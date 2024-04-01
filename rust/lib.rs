use crate::pipeline_template_generator::PipelineTemplateGenerator;
mod execution_result;
mod pipeline_template_generator;
use env_logger;
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
    job_profile_dir: PathBuf,
    microbatch_size: u32,
    tp_size: u32,
    precision: String,
    mut num_nodes: Vec<u32>,
) -> PyResult<Py<PyDict>> {
    num_nodes.sort();

    let mut generator =
        PipelineTemplateGenerator::new(job_profile_dir, microbatch_size, tp_size, precision);
    generator.divide_and_conquer(num_nodes[num_nodes.len() - 1])?;

    Python::with_gil(|py| {
        let results = PyDict::new_bound(py);

        let module = PyModule::import_bound(py, "oobleck_colossalai.pipeline_template")?;
        let class = module.getattr("PipelineTemplate")?.into_py(py);

        for num_node in num_nodes {
            let template = generator.get_pipeline_template(num_node).unwrap();
            let py_template = class
                .call1(
                    py,
                    (
                        model_name.as_str(),
                        template.get_modules_per_stage(&generator.layer_execution_results),
                        template.latency(),
                        template.mem_required(),
                    ),
                )?
                .to_object(py);
            results.set_item(template.stages.len(), py_template)?;
        }

        Ok(results.into())
    })
}

#[pymodule]
fn planner(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_function(wrap_pyfunction!(create_pipeline_templates, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::execution_result::{LayerExecutionResult, ProfileResult};
    use serde_json;
    use std::fs;
    use tempfile::TempDir;

    fn prepare(
        num_layers: u32,
        same_latency: bool,
        mut num_nodes: Vec<u32>,
    ) -> Result<(PathBuf, u32, u32, String), PlannerError> {
        let tag = "gpt2-test";
        let base_dir = TempDir::new().unwrap().path().to_path_buf();
        let path = base_dir.join(tag).join("profile");
        let microbatch_size = 1;
        let tp_size = 1;
        let precision = "fp32";
        let _ = fs::remove_dir_all(path.clone());
        fs::create_dir_all(path.clone()).unwrap();

        let profile_path = path.join(
            "profile_tp".to_string()
                + tp_size.to_string().as_str()
                + "_mb"
                + microbatch_size.to_string().as_str()
                + "_"
                + precision
                + ".json",
        );

        let mut layer_results = vec![];
        for i in 0..num_layers {
            layer_results.push(LayerExecutionResult::new(
                i,
                format!("layer{}", i),
                if same_latency {
                    1 as f64
                } else {
                    (i + 1) as f64
                },
                if same_latency {
                    1 as f64
                } else {
                    (i + 1) as f64
                },
                if same_latency {
                    1 as u64
                } else {
                    (i + 1) as u64
                },
            ));
        }

        let result = ProfileResult::new(
            "asdasd".to_string(),
            tp_size,
            microbatch_size,
            precision.to_string(),
            layer_results,
        );

        let json_string = serde_json::to_string(&result).unwrap();
        fs::write(profile_path, json_string).unwrap();

        num_nodes.sort();

        Ok((path, microbatch_size, tp_size, precision.to_string()))
    }

    #[test]
    fn test_create_pipeline_templates() {
        let num_layers = 5;
        let num_nodes = vec![1, 2, 3, 4, 5];
        let (path, microbatch_size, tp_size, precision) =
            prepare(num_layers, true, num_nodes).unwrap();

        let model_name = "gpt2".to_string();
        let num_nodes = vec![1, 2, 3, 4, 5];

        create_pipeline_templates(
            model_name,
            path,
            microbatch_size,
            tp_size,
            precision,
            num_nodes,
        )
        .unwrap();

        // let py = Python::acquire_gil();
        // let py_result = result.extract::<PyList>(py).unwrap();
        // assert_eq!(py_result.len(py), num_nodes.len());
    }
}
