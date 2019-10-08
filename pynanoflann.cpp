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
	typedef KDTreeNumpyAdaptor<f_numpy_array,num_t,DIM,Distance> self_t;
	typedef typename Distance::template traits<num_t,self_t>::distance_t metric_t;
	typedef nanoflann::KDTreeSingleIndexAdaptor< metric_t,self_t,DIM,IndexType>  index_t;

	index_t* index; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.

    std::vector<size_t> shape;
    const float *buf;

	KDTreeNumpyAdaptor(const size_t /* dimensionality */, const f_numpy_array &points, const int leaf_max_size = 10)
	{
        auto mat = points.unchecked<2>();
        buf = mat.data(0, 0);

        shape.resize(mat.ndim());
        for (ssize_t i = 0; i < mat.ndim(); i++) {
            shape[i] = static_cast<size_t>(mat.shape(i));
            std::cout << shape[i] << std::endl;
        }
		const size_t dims = 3;

		index = new index_t( static_cast<int>(dims), *this /* adaptor */, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size ) );
		index->buildIndex();
	}

	~KDTreeNumpyAdaptor() {
		delete index;
	}

	inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices, num_t *out_distances_sq, const int nChecks_IGNORED = 10) const
	{
		nanoflann::KNNResultSet<num_t,IndexType> resultSet(num_closest);
		resultSet.init(out_indices, out_distances_sq);
		index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
	}

	const self_t & derived() const {
		return *this;
	}
	self_t & derived()       {
		return *this;
	}

	inline size_t kdtree_get_point_count() const {
		return shape[0];
	}

	inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const {
        return buf[idx*3 + dim];
	}

	template <class BBOX>
	bool kdtree_get_bbox(BBOX &) const {
		return false;
	}

}; // end of KDTreeVectorOfVectorsAdaptor


class KDTree
{
public:
    KDTree(f_numpy_array);
    std::pair<f_numpy_array, i_numpy_array> query(f_numpy_array, int);
private:
    using my_kd_tree_t = KDTreeNumpyAdaptor< f_numpy_array, float >;
    my_kd_tree_t *index;
};


KDTree::KDTree(f_numpy_array points) {
    index = new my_kd_tree_t(3, points, 10);
    index->index->buildIndex();
}

std::pair<f_numpy_array, i_numpy_array> KDTree::query(f_numpy_array array, int n_neighbors) {

    std::cout << "size " << array.size() << std::endl; 

    std::vector<std::vector<size_t>> results_idxs;
    std::vector<std::vector<float>> results_dists;
    auto buf = array.request();
    float *query_data = static_cast<num_t*>(buf.ptr);
    nanoflann::KNNResultSet<num_t> resultSet(n_neighbors);

    for (size_t i = 0; i < buf.shape[0]; i++) {
        num_t *query_point = &query_data[i*3];

        std::vector<size_t> ret_indexes(n_neighbors);
        std::vector<float> out_dists_sqr(n_neighbors);

        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        index->index->findNeighbors(resultSet, query_point, nanoflann::SearchParams(10));

        results_idxs.push_back(ret_indexes);
        results_dists.push_back(out_dists_sqr);
    }

    return std::make_pair(pybind11::cast(results_dists), pybind11::cast(results_idxs));
}

PYBIND11_MODULE(pynanoflann, m) {

    pybind11::class_<KDTree>(m, "KDTree")
        .def(pybind11::init<f_numpy_array>())
        .def("query", &KDTree::query);

}
