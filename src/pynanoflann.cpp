/*
<%
setup_pybind11(cfg)
%>
*/
#include <nanoflann.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ctime>
#include <cstdlib>
#include <thread>
#include <iostream>

using namespace std;
using namespace nanoflann;

using i_numpy_array_t = pybind11::array_t<size_t, pybind11::array::c_style | pybind11::array::forcecast>;
using vvi = std::vector<std::vector<size_t>>;

template <typename num_t>
class AbstractKDTree
{
public:
    virtual void findNeighbors(nanoflann::KNNResultSet<num_t>, const num_t *query, nanoflann::SearchParams params) = 0;
    virtual size_t radiusSearch(const num_t *query, num_t radius, std::vector<std::pair<size_t, num_t>> &ret_matches, nanoflann::SearchParams params) = 0;
    virtual void knnSearch(const num_t *query, size_t num_closest, size_t *out_indices, num_t *out_distances_sq) = 0;
    virtual int saveIndex(const std::string &path) const = 0;
    virtual int loadIndex(const std::string &path) = 0;
    virtual void buildIndex() = 0;
    virtual ~AbstractKDTree() {};
};

template <typename num_t, int DIM = -1, class Distance = nanoflann::metric_L2_Simple>
struct KDTreeNumpyAdaptor : public AbstractKDTree<num_t>
{
    using self_t = KDTreeNumpyAdaptor<num_t, DIM, Distance>;
    using metric_t = typename Distance::template traits<num_t, self_t>::distance_t;
    using index_t = nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM>;
    using f_numpy_array_t = pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>;

    index_t *index;
    const num_t *buf;
    size_t n_points, dim;

    KDTreeNumpyAdaptor(const f_numpy_array_t &points, const int leaf_max_size = 10)
    {
        buf = points.template unchecked<2>().data(0, 0);
        n_points = points.shape(0);
        dim = points.shape(1);

        index = new index_t(dim, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
    }

    ~KDTreeNumpyAdaptor()
    {
        delete index;
    }
    void buildIndex()
    {
        index->buildIndex();
    }

    void findNeighbors(nanoflann::KNNResultSet<num_t> result_set, const num_t *query, nanoflann::SearchParams params)
    {
        index->findNeighbors(result_set, query, params);
    }
    void knnSearch(const num_t *query, size_t num_closest, size_t *out_indices, num_t *out_distances_sq)
    {
        index->knnSearch(query, num_closest, out_indices, out_distances_sq);
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

    int saveIndex(const std::string &path) const
    {
        FILE *f = fopen(path.c_str(), "wb");
        if (!f)
        {
            throw std::runtime_error("Error writing index file!");
        }
        index->saveIndex(f);
        int ret_val = fclose(f);
        return ret_val;
    }

    int loadIndex(const std::string &path)
    {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f)
        {
            throw std::runtime_error("Error reading index file!");
        }
        index->loadIndex(f);
        return fclose(f);
    }
};

template <typename num_t>
class KDTree
{
public:
    using f_numpy_array_t = pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>;

    KDTree(size_t n_neighbors = 10, size_t leaf_size = 10, std::string metric = "l2", float radius = 1.0f);
    void fit(f_numpy_array_t points, std::string index_path)
    {
        // Dynamic template instantiation for the popular use cases
        switch (points.shape(1))
        {
        case 1:
            if (metric == "l2")
                index = new KDTreeNumpyAdaptor<num_t, 1>(points, leaf_size);
            else
                index = new KDTreeNumpyAdaptor<num_t, 1, nanoflann::metric_L1>(points, leaf_size);
            break;
        case 2:
            if (metric == "l2")
                index = new KDTreeNumpyAdaptor<num_t, 2>(points, leaf_size);
            else
                index = new KDTreeNumpyAdaptor<num_t, 2, nanoflann::metric_L1>(points, leaf_size);
            break;
        case 3:
            if (metric == "l2")
                index = new KDTreeNumpyAdaptor<num_t, 3>(points, leaf_size);
            else
                index = new KDTreeNumpyAdaptor<num_t, 3, nanoflann::metric_L1>(points, leaf_size);
            break;
        case 4:
            if (metric == "l2")
                index = new KDTreeNumpyAdaptor<num_t, 4>(points, leaf_size);
            else
                index = new KDTreeNumpyAdaptor<num_t, 4, nanoflann::metric_L1>(points, leaf_size);
            break;
        default:
            // Arbitrary dim but works slightly slower
            if (metric == "l2")
                index = new KDTreeNumpyAdaptor<num_t, -1>(points, leaf_size);
            else
                index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L1>(points, leaf_size);
            break;
        }
        if (index_path.size())
        {
            index->loadIndex(index_path);
        }
        else
        {
            index->buildIndex();
        }
    }

    std::pair<f_numpy_array_t, i_numpy_array_t> kneighbors(f_numpy_array_t array, size_t n_neighbors)
    {
        auto mat = array.template unchecked<2>();
        const num_t *query_data = mat.data(0, 0);
        size_t n_points = mat.shape(0);
        size_t dim = mat.shape(1);

        nanoflann::KNNResultSet<num_t> resultSet(n_neighbors);
        f_numpy_array_t results_dists({n_points, n_neighbors});
        i_numpy_array_t results_idxs({n_points, n_neighbors});

        // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#direct-access
        num_t *res_dis_data = results_dists.template mutable_unchecked<2>().mutable_data(0, 0);
        size_t *res_idx_data = results_idxs.template mutable_unchecked<2>().mutable_data(0, 0);

        for (size_t i = 0; i < n_points; i++)
        {
            const num_t *query_point = &query_data[i * dim];
            resultSet.init(&res_idx_data[i * n_neighbors], &res_dis_data[i * n_neighbors]);
            index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
        }

        return std::make_pair(results_dists, results_idxs);
    }
    ~KDTree()
    {
        delete index;
    }

    std::pair<std::vector<std::vector<num_t>>, vvi> radius_neighbors(f_numpy_array_t, float radius = 1.0f);
    std::pair<f_numpy_array_t, i_numpy_array_t> kneighbors_multithreaded(f_numpy_array_t array, size_t n_neighbors, size_t nThreads=1);
    int save_index(const std::string &path);

private:
    AbstractKDTree<num_t> *index;
    size_t n_neighbors, leaf_size;
    std::string metric;
    float radius;
};

template <typename num_t>
KDTree<num_t>::KDTree(size_t n_neighbors, size_t leaf_size, std::string metric, float radius)
    : n_neighbors(n_neighbors), leaf_size(leaf_size), metric(metric), radius(radius)
{
}

template <typename num_t>
std::pair<std::vector<std::vector<num_t>>, vvi> KDTree<num_t>::radius_neighbors(f_numpy_array_t array, float radius)
{
    auto mat = array.template unchecked<2>();
    const num_t *query_data = mat.data(0, 0);
    size_t n_points = mat.shape(0);
    size_t dim = mat.shape(1);

    const num_t search_radius = static_cast<num_t>(radius);
    std::vector<std::vector<size_t>> result_idxs(n_points);
    std::vector<std::vector<num_t>> result_dists(n_points);

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

template <typename num_t>
std::pair<pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>, i_numpy_array_t> KDTree<num_t>::kneighbors_multithreaded(f_numpy_array_t array, size_t n_neighbors, size_t nThreads)
{
    auto mat = array.template unchecked<2>();
    const num_t *query_data = mat.data(0, 0);
    size_t n_points = mat.shape(0);
    size_t dim = mat.shape(1);

    f_numpy_array_t results_dists({n_points, n_neighbors});
    i_numpy_array_t results_idxs({n_points, n_neighbors});

    num_t *res_dis_data = results_dists.template mutable_unchecked<2>().mutable_data(0, 0);
    size_t *res_idx_data = results_idxs.template mutable_unchecked<2>().mutable_data(0, 0);

    auto searchBatch = [&](size_t startIdx, size_t endIdx) {
        for (size_t i = startIdx; i < endIdx; i++)
        {
            const num_t *query_point = &query_data[i * dim];
            index->knnSearch(query_point, n_neighbors, &res_idx_data[i * n_neighbors], &res_dis_data[i * n_neighbors]);
        }
    };

    std::vector<std::thread> threadPool;
    for (size_t i = 0; i < nThreads; i++)
    {
        size_t startIdx = i * (n_points / nThreads);
        size_t endIdx = (i + 1) * (n_points / nThreads);
        endIdx = std::min(endIdx, n_points);
        threadPool.push_back(std::thread(searchBatch, startIdx, endIdx));
    }
    for (auto &t : threadPool)
    {
        t.join();
    }

    return std::make_pair(results_dists, results_idxs);
}

template <typename num_t>
int KDTree<num_t>::save_index(const std::string &path)
{
    return index->saveIndex(path);
}

PYBIND11_MODULE(nanoflann_ext, m)
{
    pybind11::class_<KDTree<float>>(m, "KDTree32")
        .def(pybind11::init<size_t, size_t, std::string, float>())
        .def("fit", &KDTree<float>::fit)
        .def("kneighbors", &KDTree<float>::kneighbors)
        .def("kneighbors_multithreaded", &KDTree<float>::kneighbors_multithreaded)
        .def("radius_neighbors", &KDTree<float>::radius_neighbors)
        .def("save_index", &KDTree<float>::save_index);

    pybind11::class_<KDTree<double>>(m, "KDTree64")
        .def(pybind11::init<size_t, size_t, std::string, float>())
        .def("fit", &KDTree<double>::fit)
        .def("kneighbors", &KDTree<double>::kneighbors)
        .def("kneighbors_multithreaded", &KDTree<double>::kneighbors_multithreaded)
        .def("radius_neighbors", &KDTree<double>::radius_neighbors)
        .def("save_index", &KDTree<double>::save_index);
}
