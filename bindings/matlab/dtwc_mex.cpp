/**
 * @file dtwc_mex.cpp
 * @brief MATLAB MEX gateway for DTWC++ library.
 *
 * @details Uses the C++ MEX API (mex.hpp / mexAdapter.hpp, R2018a+).
 *          This is RAII-safe (no longjmp) and supports interleaved complex.
 *
 *          Dispatches on a command string (first argument):
 *            "dtw_distance"            -- DTW distance between two series
 *            "compute_distance_matrix" -- N x N DTW distance matrix
 *            "cluster"                 -- k-medoids clustering via Problem API
 *
 * @date 29 Mar 2026
 */

#include "mex.hpp"
#include "mexAdapter.hpp"

#include "../../dtwc/dtwc.hpp"

#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <memory>

using matlab::mex::ArgumentList;
using matlab::data::ArrayFactory;
using matlab::data::TypedArray;

class MexFunction : public matlab::mex::Function {
public:
    void operator()(ArgumentList outputs, ArgumentList inputs) override {
        // Validate: at least one argument (the command string)
        if (inputs.empty()) {
            throwError("dtwc:invalidInput",
                       "First argument must be a command string.");
        }

        const std::string cmd = getStringArg(inputs, 0);

        if (cmd == "dtw_distance") {
            cmdDtwDistance(outputs, inputs);
        } else if (cmd == "compute_distance_matrix") {
            cmdComputeDistanceMatrix(outputs, inputs);
        } else if (cmd == "cluster") {
            cmdCluster(outputs, inputs);
        } else {
            throwError("dtwc:unknownCommand",
                       "Unknown command: '" + cmd + "'. "
                       "Valid commands: dtw_distance, compute_distance_matrix, cluster.");
        }
    }

private:
    // ---------------------------------------------------------------
    //  Command: dtw_distance
    //  Inputs:  cmd, x (double vector), y (double vector),
    //           [band (int, default -1)], [metric (string, default "l1")]
    //  Outputs: scalar double distance
    // ---------------------------------------------------------------
    void cmdDtwDistance(ArgumentList &outputs, ArgumentList &inputs) {
        if (inputs.size() < 3) {
            throwError("dtwc:invalidInput",
                       "dtw_distance requires at least 2 data arguments (x, y).");
        }

        auto x = toStdVector(inputs[1]);
        auto y = toStdVector(inputs[2]);

        int band = dtwc::settings::DEFAULT_BAND_LENGTH;
        if (inputs.size() > 3) {
            band = getIntArg(inputs, 3);
        }

        // metric argument accepted for forward-compatibility but only L1 is
        // currently supported via dtwBanded
        // (future: parse inputs[4] for metric dispatch)

        double dist = dtwc::dtwBanded<double>(x, y, band);

        ArrayFactory factory;
        outputs[0] = factory.createScalar(dist);
    }

    // ---------------------------------------------------------------
    //  Command: compute_distance_matrix
    //  Inputs:  cmd, data (N x L double matrix), [band (int)]
    //  Outputs: N x N double distance matrix
    // ---------------------------------------------------------------
    void cmdComputeDistanceMatrix(ArgumentList &outputs, ArgumentList &inputs) {
        if (inputs.size() < 2) {
            throwError("dtwc:invalidInput",
                       "compute_distance_matrix requires a data matrix.");
        }

        auto series = matrixToSeriesVec(inputs[1]);
        const size_t N = series.size();

        int band = dtwc::settings::DEFAULT_BAND_LENGTH;
        if (inputs.size() > 2) {
            band = getIntArg(inputs, 2);
        }

        // Build symmetric distance matrix
        ArrayFactory factory;
        auto result = factory.createArray<double>({N, N});

        for (size_t i = 0; i < N; ++i) {
            result[i][i] = 0.0;
            for (size_t j = i + 1; j < N; ++j) {
                double d = dtwc::dtwBanded<double>(series[i], series[j], band);
                result[i][j] = d;
                result[j][i] = d;
            }
        }

        outputs[0] = std::move(result);
    }

    // ---------------------------------------------------------------
    //  Command: cluster
    //  Inputs:  cmd, data (N x L), k (int), [band (int)],
    //           [metric (string)], [maxIter (int)]
    //  Outputs: labels (1 x N int32), medoid_indices (1 x k int32),
    //           total_cost (scalar double)
    // ---------------------------------------------------------------
    void cmdCluster(ArgumentList &outputs, ArgumentList &inputs) {
        if (inputs.size() < 3) {
            throwError("dtwc:invalidInput",
                       "cluster requires data matrix and k.");
        }

        auto series = matrixToSeriesVec(inputs[1]);
        int k = getIntArg(inputs, 2);

        int band = dtwc::settings::DEFAULT_BAND_LENGTH;
        if (inputs.size() > 3) {
            band = getIntArg(inputs, 3);
        }

        // metric argument (index 4) reserved for future use

        int maxIter = dtwc::settings::DEFAULT_MAX_ITER;
        if (inputs.size() > 5) {
            maxIter = getIntArg(inputs, 5);
        }

        // Build a Problem object with the provided data
        dtwc::Problem prob("matlab_clustering");
        prob.band = band;
        prob.maxIter = maxIter;

        // Construct Data from series vectors
        const size_t N = series.size();
        std::vector<std::string> names(N);
        for (size_t i = 0; i < N; ++i) {
            names[i] = std::to_string(i);
        }
        dtwc::Data data(std::move(series), std::move(names));
        prob.set_data(std::move(data));
        prob.set_numberOfClusters(k);

        // Run clustering
        prob.fillDistanceMatrix();
        prob.cluster();

        // Extract results
        ArrayFactory factory;

        // Labels: 1-based for MATLAB convention
        auto labels = factory.createArray<int32_t>({1, N});
        for (size_t i = 0; i < N; ++i) {
            labels[0][i] = static_cast<int32_t>(prob.clusters_ind[i] + 1);
        }
        outputs[0] = std::move(labels);

        // Medoid indices: 1-based
        if (outputs.size() > 1) {
            size_t nMedoids = prob.centroids_ind.size();
            auto medoids = factory.createArray<int32_t>({1, nMedoids});
            for (size_t i = 0; i < nMedoids; ++i) {
                medoids[0][i] = static_cast<int32_t>(prob.centroids_ind[i] + 1);
            }
            outputs[1] = std::move(medoids);
        }

        // Total cost
        if (outputs.size() > 2) {
            double cost = prob.findTotalCost();
            outputs[2] = factory.createScalar(cost);
        }
    }

    // ---------------------------------------------------------------
    //  Helpers
    // ---------------------------------------------------------------

    /// Extract a string from a MATLAB char array argument.
    std::string getStringArg(ArgumentList &inputs, size_t idx) {
        if (inputs[idx].getType() != matlab::data::ArrayType::MATLAB_STRING &&
            inputs[idx].getType() != matlab::data::ArrayType::CHAR) {
            throwError("dtwc:invalidInput",
                       "Argument " + std::to_string(idx + 1) + " must be a string.");
        }

        if (inputs[idx].getType() == matlab::data::ArrayType::MATLAB_STRING) {
            matlab::data::StringArray sa = inputs[idx];
            return std::string(sa[0]);
        }

        // CHAR array
        matlab::data::CharArray ca = inputs[idx];
        return ca.toAscii();
    }

    /// Extract a scalar integer from a MATLAB numeric argument.
    int getIntArg(ArgumentList &inputs, size_t idx) {
        TypedArray<double> arr = inputs[idx];
        return static_cast<int>(arr[0]);
    }

    /// Convert a MATLAB double vector/row to std::vector<double>.
    std::vector<double> toStdVector(const matlab::data::Array &arg) {
        TypedArray<double> arr = arg;
        std::vector<double> v;
        v.reserve(arr.getNumberOfElements());
        for (auto elem : arr) {
            v.push_back(elem);
        }
        return v;
    }

    /// Convert an N x L MATLAB matrix to a vector of series (each row is one series).
    /// MATLAB stores column-major, so we iterate rows explicitly.
    std::vector<std::vector<double>> matrixToSeriesVec(const matlab::data::Array &arg) {
        TypedArray<double> mat = arg;
        auto dims = mat.getDimensions();
        size_t N = dims[0]; // number of series (rows)
        size_t L = dims[1]; // length of each series (columns)

        std::vector<std::vector<double>> series(N, std::vector<double>(L));
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < L; ++j) {
                series[i][j] = mat[i][j];
            }
        }
        return series;
    }

    /// Report an error back to MATLAB using the engine.
    void throwError(const std::string &id, const std::string &msg) {
        std::shared_ptr<matlab::engine::MATLABEngine> engine = getEngine();
        ArrayFactory factory;
        engine->feval(u"error",
                      0,
                      std::vector<matlab::data::Array>{
                          factory.createScalar(id),
                          factory.createScalar(msg)});
    }
};
