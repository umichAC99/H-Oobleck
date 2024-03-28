use crate::pipeline_template_generator::PipelineTemplateGenerator;
mod execution_result;
mod pipeline_template_generator;
use env_logger;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::fmt;
use std::path::PathBuf;

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
    microbatch_size: u32,
    mut num_nodes: Vec<u32>,
    job_profile_dir: PathBuf,
) -> PyResult<PyObject> {
    num_nodes.sort();

    let mut generator = PipelineTemplateGenerator::new(microbatch_size, job_profile_dir);
    generator.divide_and_conquer(num_nodes[num_nodes.len() - 1])?;

    Python::with_gil(|py| {
        let mut results: Vec<PyObject> = Vec::new();

        let module = PyModule::import_bound(py, "oobleck_colossalai.pipeline_template")?;
        let class = module.getattr("PipelineTemplate")?.into_py(py);

        for num_node in num_nodes {
            let template = generator.get_pipeline_template(num_node).unwrap();
            let py_template = class.call1(
                py,
                (
                    model_name.as_str(),
                    template.get_modules_per_stage(&generator.layer_execution_results),
                    template.latency(),
                    template.mem_required(),
                ),
            )?;
            results.push(py_template.to_object(py));
        }

        Ok(PyList::new_bound(py, results).to_object(py))
    })
}

#[pymodule]
fn planner(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_function(wrap_pyfunction!(create_pipeline_templates, m)?)?;
    Ok(())
}
