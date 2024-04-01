#include "execution_result.h"
#include "pipeline_template.h"
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>

namespace py = pybind11;
using namespace oobleck;

PYBIND11_MODULE(pipeline_template, m) {
  m.doc() = "Oobleck pipeline template module";

  py::class_<LayerExecutionResult>(m, "LayerExecutionResult")
      .def(
          py::init<const int, const double, const double,
                   const std::map<int, double> &, const std::map<int, double> &,
                   const std::tuple<int, int> &>(),
          py::arg("layer_index"), py::arg("forward"), py::arg("backward"),
          py::arg("allreduce_in_node"), py::arg("allreduce_across_nodes"),
          py::arg("mem_required"))
      .def_readonly("_index", &LayerExecutionResult::layer_index_)
      .def_readonly("_forward", &LayerExecutionResult::forward_)
      .def_readonly("_backward", &LayerExecutionResult::backward_)
      .def_readonly("_allreduce_in_node",
                    &LayerExecutionResult::allreduce_in_node_)
      .def_readonly("_allreduce_across_nodes",
                    &LayerExecutionResult::allreduce_across_nodes_)
      .def_readonly("_mem_required", &LayerExecutionResult::mem_required_);

  py::class_<LayerExecutionResults, std::shared_ptr<LayerExecutionResults>>(
      m, "LayerExecutionResults")
      .def(py::init<std::vector<LayerExecutionResult> &&>())
      .def("get", &LayerExecutionResults::get)
      .def("at", &LayerExecutionResults::at, py::arg("index"))
      .def_property_readonly("size", &LayerExecutionResults::size)
      .def("__repr__", [](const LayerExecutionResults &lers) {
        std::string repr = "<oobleck.LayerExecutionResults.[";
        for (const auto &ler : lers.get()) {
          repr += ler.to_string() + "\n";
        }
        repr.pop_back();
        repr.pop_back();
        repr += "]>";
        return repr;
      });

  py::class_<StageExecutionResult, std::shared_ptr<StageExecutionResult>>(
      m, "StageExecutionResult")
      .def(py::init<const std::shared_ptr<LayerExecutionResults>,
                    const std::tuple<int, int> &, const int, const int>())
      .def_readonly("_num_gpus", &StageExecutionResult::num_gpus_)
      .def_readonly("_layer_indices", &StageExecutionResult::layer_indices_)
      .def_readonly("_forward", &StageExecutionResult::forward_)
      .def_readonly("_backward", &StageExecutionResult::backward_)
      .def_property_readonly("_num_layers", &StageExecutionResult::num_layers)
      .def_readonly("_mem_required", &StageExecutionResult::mem_required_)
      .def_readonly("_node_type_idx", &StageExecutionResult::node_type_idx_)
      .def("__repr__", [](const StageExecutionResult &ser) {
        std::string repr = "<oobleck.StageExecutionResult.[";
        repr += "forward: " + std::to_string(ser.forward_) + ", ";
        repr += "backward: " + std::to_string(ser.backward_) + ", ";
        repr += "num_gpus: " + std::to_string(ser.num_gpus_) + ", ";
        repr += "layer_indices: ";
        for (const auto &index : ser.layer_indices_) {
          repr += std::to_string(index) + "-";
        }
        repr.pop_back();
        repr += "node_type_idx: " + std::to_string(ser.node_type_idx_);
        return repr;
      });

  py::class_<PipelineTemplate>(m, "PipelineTemplate")
      .def(py::init<const std::vector<std::shared_ptr<StageExecutionResult>> &,
                    const double, const int, const int, const int>())
      .def("get_stages", &PipelineTemplate::get_stages)
      .def("get_rank_grid", &PipelineTemplate::get_rank_grid, py::arg("ranks"))
      .def_property_readonly("_iteration_time",
                             &PipelineTemplate::get_iteration_time)
      .def_property_readonly("_num_nodes", &PipelineTemplate::get_num_nodes)
      .def_property_readonly("_num_gpus_per_node",
                             &PipelineTemplate::get_num_gpus_per_node)
      .def("__repr__", [](const PipelineTemplate &pt) {
        return "<oobleck.PipelineTemplate." +
               std::to_string(pt.get_num_nodes()) + "nodes>";
      });

  py::class_<NodeConfig>(m, "NodeConfig")
      .def(py::init<const std::string, const int, const int, const double>(),
           py::arg("node_type"), py::arg("num_nodes"),
           py::arg("num_gpus_per_node"), py::arg("compute_power"))
      .def_readonly("_node_type_idx", &NodeConfig::node_type_idx)
      .def_readonly("_num_nodes", &NodeConfig::num_nodes)
      .def_readonly("_num_gpus", &NodeConfig::num_gpus)
      .def_readonly("_compute_power", &NodeConfig::compute_power)
      .def("__repr__", [](const NodeConfig &sns) {
        return "<oobleck.NodeConfig." +
               node_specs[sns.node_type_idx].node_type + "[ " +
               std::to_string(sns.num_nodes) + "nodes]>";
      });

  py::class_<HeteroNodeSpec>(m, "HeteroNodeSpec")
      .def(py::init<const std::vector<NodeConfig> &>())
      .def_readonly("_node_specs", &HeteroNodeSpec::node_specs)
      .def("__repr__", [](const HeteroNodeSpec &hns) {
        std::string repr = "<oobleck.HeteroNodeSpec.[";
        for (const auto &sns : hns.node_specs) {
          repr += node_specs[sns.node_type_idx].node_type + "[" +
                  std::to_string(sns.num_nodes) + "nodes],";
        }
        repr.pop_back();
        repr.pop_back();
        repr += "]>";
        return repr;
      });

  py::class_<HeteroPipelineTemplate>(m, "HeteroPipelineTemplate")
      .def(py::init<const std::vector<std::shared_ptr<StageExecutionResult>> &,
                    const double,
                    const int, const HeteroNodeSpec &>())
      .def("get_stages", &HeteroPipelineTemplate::get_stages)
      .def("get_node_spec", &HeteroPipelineTemplate::get_node_spec)
      .def_property_readonly("_iteration_time",
                             &HeteroPipelineTemplate::get_iteration_time)
      .def("__repr__", [](const HeteroPipelineTemplate &hpt) {
        std::string repr = "<oobleck.HeteroPipelineTemplate.[";
        repr += "t: " + std::to_string(hpt.get_iteration_time()) + ", ";
        repr += "stages: [";
        for (const auto &stage : hpt.get_stages()) {
          repr += stage->to_string() + ", ";
        }
        repr.pop_back();
        repr.pop_back();
        repr += "], ";
        repr += "]>";
        return repr;
      });

  py::class_<PipelineTemplateGenerator>(m, "PipelineTemplateGenerator")
      .def(py::init<>())
      .def("create_pipeline_templates",
           &PipelineTemplateGenerator::create_pipeline_templates)
      .def("create_hetero_pipeline_template",
           &PipelineTemplateGenerator::create_hetero_pipeline_template);

  m.def("get_profile_results", &get_profile_results, py::arg("model_name"),
        py::arg("model_tag"), py::arg("microbatch_size"), py::arg("node_type"));
}