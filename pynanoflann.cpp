/*
<%
setup_pybind11(cfg)
cfg['include_dirs'] = ['/home/u1234x1234/libs/nanoflann/include/']
%>
*/
#include <nanoflann.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <typeinfo>

using namespace std;
using namespace nanoflann;

using num_t = float;
using f_numpy_array = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>;
using i_numpy_array = pybind11::array_t<size_t, pybind11::array::c_style | pybind11::array::forcecast>;

using MM = f_numpy_array;

template <class MM, typename num_t = float, int DIM = -1, class Distance = nanoflann::metric_L2_Simple, typename IndexType = size_t>
struct KDTreeNumpyAdaptor
{
    typedef KDTreeNumpyAdaptor<f_numpy_array, num_t, DIM, Distance> self_t;
    typedef typename Distance::template traits<num_t, self_t>::distance_t metric_t;
    typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType> index_t;

    index_t *index;
    std::vector<size_t> shape;
    const float *buf;

    KDTreeNumpyAdaptor(const f_numpy_array &points, const int leaf_max_size = 10)
    {
        auto mat = points.unchecked<2>();
        buf = mat.data(0, 0);

        shape.resize(mat.ndim());
        for (ssize_t i = 0; i < mat.ndim(); i++)
        {
            shape[i] = static_cast<size_t>(mat.shape(i));
        }

        index = new index_t(shape[1], *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        index->buildIndex();
    }

    ~KDTreeNumpyAdaptor()
    {
        delete index;
    }

    inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices, num_t *out_distances_sq) const
    {
        nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }

    const self_t &derived() const
    {
        return *this;
    }
    self_t &derived()
    {
        return *this;
    }

    inline size_t kdtree_get_point_count() const
    {
        return shape[0];
    }

    inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return buf[idx * shape[1] + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const
    {
        return false;
    }
};

class __attribute__((visibility("hidden"))) KDTree
{
public:
    KDTree(f_numpy_array);
    std::pair<f_numpy_array, i_numpy_array> query(f_numpy_array, size_t);

private:
    using my_kd_tree_t = KDTreeNumpyAdaptor<f_numpy_array, num_t>;
    my_kd_tree_t *index;
};

KDTree::KDTree(f_numpy_array points)
{
    index = new my_kd_tree_t(points, 10);
    index->index->buildIndex();
}

std::pair<f_numpy_array, i_numpy_array> KDTree::query(f_numpy_array array, size_t n_neighbors)
{
    auto mat = array.unchecked<2>();
    const num_t *query_data = mat.data(0, 0);
    size_t nPoints = mat.shape(0);
    size_t dim = mat.shape(1);

    nanoflann::KNNResultSet<num_t> resultSet(n_neighbors);
    f_numpy_array results_dists({nPoints, n_neighbors});
    i_numpy_array results_idxs({nPoints, n_neighbors});

    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#direct-access
    num_t *res_dis_data = results_dists.mutable_unchecked<2>().mutable_data(0, 0);
    size_t *res_idx_data = results_idxs.mutable_unchecked<2>().mutable_data(0, 0);

    for (size_t i = 0; i < nPoints; i++)
    {
        const num_t *query_point = &query_data[i * dim];
        resultSet.init(&res_idx_data[i*n_neighbors], &res_dis_data[i*n_neighbors]);
        index->index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }

    return std::make_pair(results_dists, results_idxs);
}

PYBIND11_MODULE(pynanoflann, m)
{
    pybind11::class_<KDTree>(m, "KDTree")
        .def(pybind11::init<f_numpy_array>())
        .def("query", &KDTree::query);
}
