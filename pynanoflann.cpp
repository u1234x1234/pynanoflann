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

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <typeinfo>

using namespace std;
using namespace nanoflann;

using num_t = float;
using f_numpy_array_t = pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>;
using i_numpy_array_t = pybind11::array_t<size_t, pybind11::array::c_style | pybind11::array::forcecast>;

using vvi = std::vector<std::vector<size_t>>;
using vvf = std::vector<std::vector<float>>;

class AbstractKDTree
{
public:
    virtual void findNeighbors(nanoflann::KNNResultSet<num_t>, const num_t *query, nanoflann::SearchParams params) = 0;
    virtual size_t radiusSearch(const num_t *query, num_t radius, std::vector<std::pair<size_t, num_t>> &ret_matches, nanoflann::SearchParams params) = 0;
};

template <typename num_t = float, int DIM = -1, class Distance = nanoflann::metric_L2_Simple>
struct KDTreeNumpyAdaptor : public AbstractKDTree
{
    using self_t = KDTreeNumpyAdaptor<num_t, DIM, Distance>;
    using metric_t = typename Distance::template traits<num_t, self_t>::distance_t;
    using index_t = nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM>;

    index_t *index;
    const float *buf;
    size_t n_points, dim;

    KDTreeNumpyAdaptor(const f_numpy_array_t &points, const int leaf_max_size = 10)
    {
        buf = points.unchecked<2>().data(0, 0);
        n_points = points.shape(0);
        dim = points.shape(1);

        index = new index_t(dim, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        index->buildIndex();
    }

    ~KDTreeNumpyAdaptor()
    {
        delete index;
    }

    void findNeighbors(nanoflann::KNNResultSet<num_t> result_set, const num_t *query, nanoflann::SearchParams params)
    {
        index->findNeighbors(result_set, query, params);
    }

    size_t radiusSearch(const num_t *query, num_t radius, std::vector<std::pair<size_t, num_t>> &ret_matches, nanoflann::SearchParams params)
    {
        return index->radiusSearch(query, radius, ret_matches, params);
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
        return n_points;
    }

    inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return buf[idx * this->dim + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const
    {
        return false;
    }
};

class KDTree
{
public:
    KDTree(size_t n_neighbors = 10, size_t leaf_size = 10, std::string metric = "l2", float radius = 1.0f);
    void fit(f_numpy_array_t);
    std::pair<f_numpy_array_t, i_numpy_array_t> kneighbors(f_numpy_array_t, size_t);
    std::pair<vvf, vvi> radius_neighbors(f_numpy_array_t, float radius = 1.0f);

private:
    AbstractKDTree *index;
    size_t n_neighbors, leaf_size;
    std::string metric;
    float radius;
};

KDTree::KDTree(size_t n_neighbors, size_t leaf_size, std::string metric, float radius)
    : n_neighbors(n_neighbors), leaf_size(leaf_size), metric(metric), radius(radius)
{
}

void KDTree::fit(f_numpy_array_t points)
{
    // Dynamic template instantiation for the popular use cases
    switch (points.shape(1))
    {
    case 1:
        if (metric == "l2")
            index = new KDTreeNumpyAdaptor<float, 1>(points, leaf_size);
        else
            index = new KDTreeNumpyAdaptor<float, 1, nanoflann::metric_L1>(points, leaf_size);
        break;
    case 2:
        if (metric == "l2")
            index = new KDTreeNumpyAdaptor<float, 2>(points, leaf_size);
        else
            index = new KDTreeNumpyAdaptor<float, 2, nanoflann::metric_L1>(points, leaf_size);
        break;
    case 3:
        if (metric == "l2")
            index = new KDTreeNumpyAdaptor<float, 3>(points, leaf_size);
        else
            index = new KDTreeNumpyAdaptor<float, 3, nanoflann::metric_L1>(points, leaf_size);
        break;
    case 4:
        if (metric == "l2")
            index = new KDTreeNumpyAdaptor<float, 4>(points, leaf_size);
        else
            index = new KDTreeNumpyAdaptor<float, 4, nanoflann::metric_L1>(points, leaf_size);
        break;
    default:
        if (metric == "l2")
            index = new KDTreeNumpyAdaptor<float, -1>(points, leaf_size);
        else
            index = new KDTreeNumpyAdaptor<float, -1, nanoflann::metric_L1>(points, leaf_size);
        break;
    }
}

std::pair<f_numpy_array_t, i_numpy_array_t> KDTree::kneighbors(f_numpy_array_t array, size_t n_neighbors)
{
    auto mat = array.unchecked<2>();
    const num_t *query_data = mat.data(0, 0);
    size_t n_points = mat.shape(0);
    size_t dim = mat.shape(1);

    nanoflann::KNNResultSet<num_t> resultSet(n_neighbors);
    f_numpy_array_t results_dists({n_points, n_neighbors});
    i_numpy_array_t results_idxs({n_points, n_neighbors});

    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#direct-access
    num_t *res_dis_data = results_dists.mutable_unchecked<2>().mutable_data(0, 0);
    size_t *res_idx_data = results_idxs.mutable_unchecked<2>().mutable_data(0, 0);

    for (size_t i = 0; i < n_points; i++)
    {
        const num_t *query_point = &query_data[i * dim];
        resultSet.init(&res_idx_data[i * n_neighbors], &res_dis_data[i * n_neighbors]);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }

    return std::make_pair(results_dists, results_idxs);
}

std::pair<vvf, vvi> KDTree::radius_neighbors(f_numpy_array_t array, float radius)
{
    auto mat = array.unchecked<2>();
    const num_t *query_data = mat.data(0, 0);
    size_t n_points = mat.shape(0);
    size_t dim = mat.shape(1);

    const num_t search_radius = static_cast<num_t>(radius);
    std::vector<std::vector<size_t>> result_idxs(n_points);
    std::vector<std::vector<float>> result_dists(n_points);

    std::vector<std::pair<size_t, num_t>> ret_matches;

    for (size_t i = 0; i < n_points; i++)
    {
        const num_t *query_point = &query_data[i * dim];
        const size_t nMatches = index->radiusSearch(&query_point[0], search_radius, ret_matches, nanoflann::SearchParams());
        for (size_t j = 0; j < nMatches; j++)
        {
            result_idxs[i].push_back(ret_matches[j].first);
            result_dists[i].push_back(ret_matches[j].second);
        }
    }
    // TODO Copy will be made
    return std::make_pair(result_dists, result_idxs);
}

PYBIND11_MODULE(pynanoflann, m)
{
    pybind11::class_<KDTree>(m, "KDTree")
        .def(pybind11::init<size_t, size_t, std::string, float>())
        .def("fit", &KDTree::fit)
        .def("kneighbors", &KDTree::kneighbors)
        .def("radius_neighbors", &KDTree::radius_neighbors);
}
