#include "pipeline_template.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cppcoro/when_all.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <ranges>
#include <string>
#include <map>

#ifdef PYBIND11_MODULE
#include <pybind11/pybind11.h>
#endif

// #define DEBUG

/**
 * Extension of Section 4.1.2. GPU-Stage Mapping using divide and conquer
 * algorithm for heterogenous system. The divide and conquer is accelerated
 * using multithreading and memoization.
 */

namespace oobleck {

std::vector<SingleNodeSpec> node_specs;
std::map<std::string, int> node_specs_map;

std::string HeteroNodeSpec::get_cache_key() const {
  std::string result = "";
  for (auto &config : node_specs) {
    result += DCExecutionResult::get_device_indices_key(
                  config.num_nodes, config.num_gpus, config.node_type_idx) +
              "-";
  }
  result.pop_back();
  return result;
}

void generateSubsetsUtil(const HeteroNodeSpec &originalSpec,
                         std::vector<HeteroNodeSpec> &allSubsets,
                         HeteroNodeSpec currentSubset, int start) {
  for (int i = start; i < originalSpec.node_specs.size(); ++i) {
    for (int j = 1; j <= originalSpec.node_specs[i].num_nodes; ++j) {
      HeteroNodeSpec newSubset = currentSubset;
      newSubset.node_specs[i].num_nodes = j;
      newSubset.update_fields();
      // skip the original set
      if (newSubset.num_total_nodes != originalSpec.num_total_nodes) {
        allSubsets.push_back(newSubset);
      }
      generateSubsetsUtil(originalSpec, allSubsets, newSubset, i + 1);
    }
  }
}


// enumerate all possible subsets of the original cluster set
std::vector<HeteroNodeSpec> generateSubsets(const HeteroNodeSpec &heteroSpec) {
  std::vector<HeteroNodeSpec> allSubsets;
  HeteroNodeSpec currentSubset = heteroSpec;
  for (int i = 0; i < heteroSpec.node_specs.size(); ++i) {
    currentSubset.node_specs[i].num_nodes = 0;
  }
  generateSubsetsUtil(heteroSpec, allSubsets, currentSubset, 0);

#ifdef DEBUG
  // Print all subsets for demonstration purposes
  for (const auto &subset : allSubsets) {
    std::cout << subset.to_string() << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Total number of subsets: " << allSubsets.size() << std::endl;

  // Check there are no duplicates
  for (int i = 0; i < allSubsets.size(); ++i) {
    for (int j = i + 1; j < allSubsets.size(); ++j) {
      assert(!(allSubsets[i].node_specs == allSubsets[j].node_specs));
    }
  }
#endif
  return allSubsets;
}

std::vector<std::shared_ptr<LayerExecutionResults>>
get_hetero_profile_results(const std::vector<std::string> &model_names,
                           const std::vector<std::string> &model_tags,
                           const int microbatch_size,
                           const std::vector<std::string> &node_types) {
  std::vector<std::shared_ptr<LayerExecutionResults>> layer_execution_results;
  for (int i = 0; i < node_types.size(); i++) {
    layer_execution_results.push_back(get_profile_results(
        model_names[i], model_tags[i], microbatch_size, node_types[i]));
  }
  return layer_execution_results;
} // get_hetero_profile_results

HeteroPipelineTemplate
PipelineTemplateGenerator::create_hetero_pipeline_template(
    std::vector<std::shared_ptr<LayerExecutionResults>> layer_execution_results,
    const HeteroNodeSpec &node_spec) {
#ifdef PYBIND11_MODULE
  // Release GIL
  pybind11::gil_scoped_release release;
#endif

  std::cout << "layer_execution_results.size(): "
            << layer_execution_results.size() << std::endl;
  std::cout << "node_spec.to_string(): " << node_spec.to_string() << std::endl;
  // assert the number of node specs matched the layer execution results length
  assert(layer_execution_results.size() == node_spec.node_specs.size());

  // assert that the number of layers is the same for node types
  for (int i = 1; i < layer_execution_results.size(); i++) {
    assert(layer_execution_results[i]->size() ==
           layer_execution_results[i - 1]->size());
  }

  // iterate through all total number of stages
  const int layer_count = layer_execution_results[0]->size();
  const int min_num_stages = node_spec.num_total_nodes;
  const int max_num_stages = layer_count;

  std::cout << "min_num_stages: " << min_num_stages << std::endl;
  std::cout << "max_num_stages: " << max_num_stages << std::endl;

  std::vector<cppcoro::task<std::shared_ptr<DCExecutionResult>>>
      num_stages_tasks;
  for (int num_stages = min_num_stages; num_stages <= max_num_stages;
       num_stages++) {
    num_stages_tasks.emplace_back(divide_and_conquer(
        layer_execution_results, std::make_tuple(0, layer_count), num_stages,
        node_spec));
  }

  std::cout << "Waiting for tasks" << std::endl;
  std::vector<std::shared_ptr<DCExecutionResult>> results =
      cppcoro::sync_wait(cppcoro::when_all(std::move(num_stages_tasks)));
  std::cout << "Wait done" << std::endl;

  std::cout << "Cache hit: " << cache_hit_.load()
            << ", miss: " << cache_miss_.load() << std::endl;

  if (std::all_of(results.begin(), results.end(),
                  [](const std::shared_ptr<DCExecutionResult> &result)
                      -> bool { return result == nullptr; })) {
    std::cout << "All results are invalid" << std::endl;
  }

  auto optimal_result = [&]() -> std::shared_ptr<DCExecutionResult> {
      std::shared_ptr<DCExecutionResult> result(nullptr);
      for (int i = 0; i < results.size(); i++) {
        if (result == nullptr) {
          result = results[i];
        } else if (results[i] != nullptr &&
                   results[i]->get_t() < result->get_t()) {
          result = results[i];
        }
      }
      return result;
    }();

  assert(optimal_result != nullptr &&
           optimal_result->get_stages().size() > 0);
  return HeteroPipelineTemplate(optimal_result->get_stages(), optimal_result->get_t(),
                                layer_count,
                                node_spec);
}

cppcoro::task<std::shared_ptr<DCExecutionResult>>
PipelineTemplateGenerator::divide_and_conquer(
    const std::vector<std::shared_ptr<LayerExecutionResults>>
        &layer_execution_results,
    const std::tuple<int, int> layer_indices, const int num_stages,
    const HeteroNodeSpec &node_spec) {
  co_await thread_pool_.schedule();

  int start_layer_index = std::get<0>(layer_indices);
  int end_layer_index = std::get<1>(layer_indices);

  std::shared_ptr<DCExecutionResult> result(nullptr);

  DCExecutionResult::key key =
      std::make_tuple(num_stages, start_layer_index, end_layer_index,
                      node_spec.get_cache_key());

  // Return cached result if it exists
  auto it = dc_cache_.find(key);
  if (it != dc_cache_.end()) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    result = it->second;
    co_return result;
  }

  cache_miss_.fetch_add(1, std::memory_order_relaxed);

  // Infeasible cases
  bool infeasible = false;
  if (num_stages > end_layer_index - start_layer_index) {
    // If the number of stages is more than number of layers
    infeasible = true;
  }

  int num_total_nodes = node_spec.num_total_nodes;
  int num_gpus = -1;
  if (num_total_nodes == 1) {
    assert(node_spec.idx_to_only_node != -1 &&
           "idx_to_only_node is not set when num_total_nodes is 1");
    assert(node_spec.node_specs[node_spec.idx_to_only_node].num_nodes == 1 &&
           "num_nodes is not 1 when num_total_nodes is 1");
    num_gpus = node_spec.node_specs[node_spec.idx_to_only_node].num_gpus;
    if (num_gpus < num_stages) {
      // At least one GPU should be assigned to each stage
      infeasible = true;
    }

    double log_num_gpus = log2(num_gpus);
    if (num_stages == 1 && log_num_gpus != trunc(log_num_gpus)) {
      infeasible = true;
    }
  } else if (num_total_nodes > num_stages) {
    infeasible = true;
  }

  if (infeasible) {
    dc_cache_.insert({key, nullptr});
    // accessor->second = nullptr;
    co_return nullptr;
  }

  // Base case (conquer phase)
  if (num_stages == 1) {
    assert(num_total_nodes == 1);
    num_gpus = node_spec.node_specs[node_spec.idx_to_only_node].num_gpus;
    int node_type_idx =
        node_spec.node_specs[node_spec.idx_to_only_node].node_type_idx;
    // If there is only one stage, assign all layers to that stage
    auto stage = std::make_shared<StageExecutionResult>(
        layer_execution_results[node_type_idx], layer_indices, num_gpus,
        node_type_idx);
    auto result = std::make_shared<DCExecutionResult>(stage);
    dc_cache_.insert({key, result});
    co_return result;
  }

  // Divide phase
  for (int k : std::ranges::iota_view<int, int>(start_layer_index + 1,
                                                end_layer_index)) {
    // if (num_total_nodes == 1) {
    //   // Split GPUs in a node
    //   assert(num_gpus != -1);
    //   assert(node_spec.idx_to_only_node != -1);
    //   for (int num_gpus_left : std::ranges::iota_view<int, int>(1, num_gpus)) {
    //     // TODO: understand why
    //     if (num_gpus_left != num_gpus - num_gpus_left) {
    //       continue;
    //     }

    //     for (int num_stages_left :
    //          std::ranges::iota_view<int, int>(1, num_stages)) {
    //       std::shared_ptr<DCExecutionResult> result_left(nullptr);
    //       std::shared_ptr<DCExecutionResult> result_right(nullptr);

    //       auto node_spec_left = node_spec;
    //       node_spec_left.node_specs[node_spec.idx_to_only_node].num_gpus =
    //           num_gpus_left;
    //       auto key_left = std::make_tuple(num_stages_left, start_layer_index, k,
    //                                       node_spec_left.get_cache_key());

    //       auto it = dc_cache_.find(key_left);
    //       if (it != dc_cache_.end()) {
    //         result_left = it->second;
    //       } else {
    //         result_left = co_await divide_and_conquer(
    //             layer_execution_results, std::make_tuple(start_layer_index, k),
    //             num_stages_left, node_spec_left);
    //       }

    //       auto node_spec_right = node_spec;
    //       node_spec_right.node_specs[node_spec.idx_to_only_node].num_gpus =
    //           num_gpus - num_gpus_left;
    //       auto key_right =
    //           std::make_tuple(num_stages - num_stages_left, k, end_layer_index,
    //                           node_spec_right.get_cache_key());

    //       it = dc_cache_.find(key_right);
    //       if (it != dc_cache_.end()) {
    //         result_right = it->second;
    //       } else {
    //         result_right = co_await divide_and_conquer(
    //             layer_execution_results, std::make_tuple(k, end_layer_index),
    //             num_stages - num_stages_left, node_spec_right);
    //       }

    //       if (result_left == nullptr || result_right == nullptr) {
    //         continue;
    //       }

    //       auto new_result =
    //           std::make_shared<DCExecutionResult>(result_left, result_right);
    //       if (result == nullptr || new_result->get_t() < result->get_t()) {
    //         result = new_result;
    //       }
    //     }
    //   } // for num_gpus_left
    // }   // if num_nodes == 1
    // else {
      // Split nodes
      std::vector<HeteroNodeSpec> all_node_spec_subsets =
          generateSubsets(node_spec);
      for (auto &node_spec_subset_left : all_node_spec_subsets) {
        auto node_spec_subset_right = node_spec.subtract(node_spec_subset_left);
        // std::cout << "origin " << node_spec.to_string() << std::endl;
        // std::cout << "left " << node_spec_subset_left.to_string() << std::endl;
        // std::cout << "right " << node_spec_subset_right.to_string()
                  // << std::endl;
        for (int num_stages_left :
             std::ranges::iota_view<int, int>(1, num_stages)) {

          std::shared_ptr<DCExecutionResult> result_left(nullptr);
          std::shared_ptr<DCExecutionResult> result_right(nullptr);
          auto key_left =
              std::make_tuple(num_stages_left, start_layer_index, k,
                              node_spec_subset_left.get_cache_key());
          auto it = dc_cache_.find(key_left);
          if (it != dc_cache_.end()) {
            result_left = it->second;
          } else {
            result_left = co_await divide_and_conquer(
                layer_execution_results, std::make_tuple(start_layer_index, k),
                num_stages_left, node_spec_subset_left);
          }

          auto key_right =
              std::make_tuple(num_stages - num_stages_left, k, end_layer_index,
                              node_spec_subset_right.get_cache_key());
          it = dc_cache_.find(key_right);
          if (it != dc_cache_.end()) {
            result_right = it->second;
          } else {
            result_right = co_await divide_and_conquer(
                layer_execution_results, std::make_tuple(k, end_layer_index),
                num_stages - num_stages_left, node_spec_subset_right);
          }

          if (result_left == nullptr || result_right == nullptr) {
            continue;
          }

          auto new_result =
              std::make_shared<DCExecutionResult>(result_left, result_right);
          if (result == nullptr || new_result->get_t() < result->get_t()) {
            result = new_result;
          }
        } // for stages
      }   // for node_spec_subset
    // }     // if num_nodes != 1
  }       // divide for loop

  dc_cache_.insert({key, result});
  co_return result;
}

} // namespace oobleck