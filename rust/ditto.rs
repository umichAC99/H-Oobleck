use crate::execution_result::*;
use crate::PlannerError;
use log;
use std::rc::Rc;

struct Choice {
    num_devices: i32,
    num_stages: i32,
}

type DPChoices = Vec<Vec<Choice>>;
type DeviceResource = Vec<i32>;

struct DPState {
    devices: DeviceResource,
    execution_result: Option<Rc<PipelineExecutionResult>>,
}

pub struct ButtomUpDPPipelineRecoverSolver {
    node_folding_factor: Vec<i32>,
    dp: Vec<DPState>,
    dp_choices: DPChoices,
    avail_devices: DeviceResource,
}

impl ButtomUpDPPipelineRecoverSolver {
    pub fn new(node_folding_factor: Vec<i32>) -> Self {

        ButtomUpDPPipelineRecoverSolver {
            node_folding_factor: node_folding_factor,
            dp: vec![],
            dp_choices: vec![],
            avail_devices: vec![],
        }
    }

    pub fn solve(&mut self, cluster_spec: Vec<i32>, modules_per_stage: Vec<Vec<String>>, layers: &Vec<Vec<LayerExecutionResult>>) -> Result<(), PlannerError> {

        log::debug!("Solving pipeline recovery problem");
        log::debug!("Virtual pipeline template: {:?}", modules_per_stage);
        return Ok(());
    }
}