use crate::pipeline_template_generator::PipelineTemplateGenerator;
use crate::ditto::ButtomUpDPPipelineRecoverSolver;
mod execution_result;
mod pipeline_template_generator;
mod ditto;
use env_logger;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt;

#[derive(Debug)]
struct PlannerError {
    message: String,
}

impl PlannerError {
    fn new(message: &str) -> Self {
        PlannerError {
            message: message.to_string(),
        }
    }
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PlannerError: {}", self.message)
    }
}

impl std::error::Error for PlannerError {}

impl From<PlannerError> for PyErr {
    fn from(error: PlannerError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

#[pyfunction]
fn create_pipeline_templates(
    model_name: String,
    profile_data: Vec<execution_result::LayerExecutionResult>,
    mut num_nodes: Vec<u32>,
) -> PyResult<Py<PyDict>> {
    num_nodes.sort();

    let mut generator = PipelineTemplateGenerator::new(profile_data);
    generator.divide_and_conquer(num_nodes[num_nodes.len() - 1])?;

    Python::with_gil(|py| {
        let results = PyDict::new_bound(py);

        let module = PyModule::import_bound(py, "cornstarch.pipeline_template")?;
        let class = module.getattr("PipelineTemplate")?.into_py(py);

        for num_node in num_nodes {
            let result = generator.get_pipeline_template(num_node).unwrap();
            let py_template = class
                .call1(
                    py,
                    (
                        model_name.as_str(),
                        result.get_modules_per_stage(&generator.layer_execution_results),
                        result.latency(),
                        result.stages[result.kstar].latency(),
                        result.mem_required(),
                    ),
                )?
                .to_object(py);
            results.set_item(result.stages.len(), py_template)?;
        }

        Ok(results.into())
    })
}

#[pyfunction]
fn create_base_hetero_pipeline_template(
    model_name: String,
    profile_data: Vec<execution_result::LayerExecutionResult>,
    num_nodes: u32,
) -> PyResult<PyObject> {
    // Create a generator
    let mut generator = PipelineTemplateGenerator::new(profile_data);
    generator.divide_and_conquer(num_nodes)?;

    Python::with_gil(|py| {
        // Import the Python module and class
        let module = PyModule::import(py, "oobleck.planning.ditto")?;
        let class = module.getattr("HeteroPipelineTemplate")?;

        // Generate the pipeline template
        let result = generator.get_pipeline_template(num_nodes).unwrap();

        // Call the Python class constructor to create an instance
        let py_template = class.call1((
            model_name.as_str(),
            result.get_modules_per_stage(&generator.layer_execution_results),
            result.get_latency_per_stage(),
            result.get_device_idx_per_stage(),
            result.latency(),
            result.stages[result.kstar].latency(),
            result.mem_required(),
        ))?;

        Ok(py_template.to_object(py))
    })
}

#[pyfunction]
fn dynamic_programming_recovery(
    node_folding_factor: Vec<i32>,
    cluster_spec: Vec<i32>,
    virtual_stages: Vec<Vec<i32>>,
    layers: Vec<Vec<execution_result::LayerExecutionResult>>,
) -> PyResult<PyObject> {
    println!("Received node_folding_factor: {:?}", node_folding_factor);
    println!("Received cluster_spec: {:?}", cluster_spec);
    let mut solver = ButtomUpDPPipelineRecoverSolver::new(node_folding_factor);
    let result = solver.solve(cluster_spec, &virtual_stages, &layers).unwrap();

    Python::with_gil(|py| {
        let module = PyModule::import(py, "oobleck.planning.ditto")?;
        let class = module.getattr("HeteroPipelineTemplate")?;

        let py_template = class.call1(
            (
            "gpt2",
            result.get_modules_per_stage(&layers[0]),
            result.get_latency_per_stage(),
            result.get_device_idx_per_stage(),
            result.latency(),
            result.stages[result.kstar].latency(),
            result.mem_required(),
            )
        )?;

        Ok(py_template.to_object(py))
    })

}


#[pymodule]
fn planner(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_function(wrap_pyfunction!(create_pipeline_templates, m)?)?;
    m.add_function(wrap_pyfunction!(create_base_hetero_pipeline_template, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_programming_recovery, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::execution_result::LayerExecutionResult;

    fn prepare(
        num_layers: u32,
        same_latency: bool,
        mut num_nodes: Vec<u32>,
    ) -> Vec<LayerExecutionResult> {
        let mut layer_results = vec![];
        for i in 0..num_layers {
            layer_results.push(LayerExecutionResult {
                layer_index: i,
                layer_name: format!("layer{}", i),
                forward: if same_latency {
                    1 as f64
                } else {
                    (i + 1) as f64
                },
                backward: if same_latency {
                    1 as f64
                } else {
                    (i + 1) as f64
                },
                mem_required: if same_latency {
                    1 as u64
                } else {
                    (i + 1) as u64
                },
            });
        }

        num_nodes.sort();

        layer_results
    }

    #[test]
    fn test_create_pipeline_templates() {
        let num_layers = 5;
        let num_nodes = vec![1, 2, 3, 4, 5];
        let layer_results = prepare(num_layers, true, num_nodes.clone());

        let model_name = "gpt2".to_string();

        create_pipeline_templates(model_name, layer_results, num_nodes).unwrap();

        // let py = Python::acquire_gil();
        // let py_result = result.extract::<PyList>(py).unwrap();
        // assert_eq!(py_result.len(py), num_nodes.len());
    }
}
