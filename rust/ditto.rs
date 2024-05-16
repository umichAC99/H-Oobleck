use crate::execution_result::*;
use crate::PlannerError;
use log;
use std::rc::Rc;
use std::sync::Arc;

// Choice is a struct that represents the choice of using a number of devices
// to map to a number of stages.
// e.g. num_devices = 2, num_stages = 3 means that 2 devices are used to map 3 stages.
#[derive(Debug, Clone)]
struct Choice {
    num_devices: i32,
    num_stages: i32,
}

// DPChoices is a 2D vector that represents all availiable choices for different device types
// e.g. [[(2, 4)], [(2, 2)]] means that for first type of device, we can use 2 devices to map 4 stages
//                                      for second type of device, we can use 2 devices to map 2 stages
type DPChoices = Vec<Vec<Choice>>;

// DeviceResource is a vector that represents the number of devices of each type
// e.g. [1, 2, 3] means that there are 3 types of devices, and the number of devices for each type is 1, 2, 3
type DeviceResource = Vec<i32>;

// DPState is a struct that represents the state of the dynamic programming algorithm
#[derive(Debug, Clone)]
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

    // Merge stages from start_stage_idx to end_stage_idx
    fn merge_stage(&self,virtual_stages: &Vec<Vec<i32>>, start_stage_idx: usize, end_stage_idx: usize, layers: &Vec<LayerExecutionResult>, device_type_idx: u32) -> Arc<StageExecutionResult> {
        let left_layer_idx = virtual_stages[start_stage_idx][0] as usize;
        let right_layer_idx = virtual_stages[end_stage_idx][1] as usize;
        return Arc::new(StageExecutionResult::new_with_device_type(&layers[left_layer_idx..=right_layer_idx], device_type_idx));
    }

    fn pretty_print(&self) {
        log::debug!("DP State:");
        for i in 0..self.dp.len() {
            log::debug!("DP State {}: {:?}", i, self.dp[i]);
        }
    }

    fn preprocess(&mut self, cluster_spec: Vec<i32>, virtual_stages: &Vec<Vec<i32>>, layers: &Vec<Vec<LayerExecutionResult>>) -> Result<(), PlannerError> {
        // Initialize avail_devices
        self.avail_devices = cluster_spec.clone();

        // Initialize DP
        self.dp = vec![DPState {
            devices: vec![0; cluster_spec.len()],
            execution_result: None,
        }; virtual_stages.len()+1];

        // Initialize all possible dp_choices_ for different node types
        self.dp_choices = vec![vec![]; cluster_spec.len()];
        for i in 0..cluster_spec.len() {
            self.dp_choices[i].push(Choice {
                num_devices: 1,
                num_stages: self.node_folding_factor[i],
            });

            let pipeline_execution_result =
                    PipelineExecutionResult::make_base_result(self.merge_stage(virtual_stages, 0, self.node_folding_factor[i] as usize, &layers[i], i as u32));
            self.dp[self.node_folding_factor[i] as usize].execution_result = Some(Rc::new(pipeline_execution_result));
            self.dp[self.node_folding_factor[i] as usize].devices[i] = 1;
        }
        return Ok(());
    }

    pub fn solve(&mut self, cluster_spec: Vec<i32>, virtual_stages: &Vec<Vec<i32>>, layers: &Vec<Vec<LayerExecutionResult>>) -> Result<PipelineExecutionResult, PlannerError> {

        log::debug!("Solving pipeline recovery problem");
        log::debug!("Virtual pipeline template: {:?}", virtual_stages);

        // if length of cluster_spec does not match length of layers, return error
        if cluster_spec.len() != layers.len() {
            return Err(PlannerError::new(
                format!("length of cluster_spec does not match length of profiles").as_str(),
            ))?
        }

        // preprocess DP states
        self.preprocess(cluster_spec, virtual_stages, layers)?;
        self.pretty_print();

        // start DP algorithm
        for i in 1..=(self.dp.len()){
            for node_type_idx in 0..(self.dp_choices.len()){
                for choice in self.dp_choices[node_type_idx].iter(){
                    let num_devices = choice.num_devices;
                    let num_stages = choice.num_stages as usize;

                    if i < num_stages {
                        continue;
                    }
                    if self.dp[i - num_stages].execution_result.is_none() {
                        continue;
                    }
                    // if overuse devices, continue
                    if self.dp[i - num_stages].devices[node_type_idx] + num_devices > self.avail_devices[node_type_idx] {
                        continue;
                    }
                    
                    /*
                        dp[i] = 
                            min(dp[i], merge_results(
                                dp[i - choice.stages], 
                                result([i - choice.stages...i], choice.devices
                                )
                                )
                            )
                    */
                    let mut new_dp_state = self.dp[i - num_stages].clone();
                    new_dp_state.devices[node_type_idx] += num_devices;
                    let new_pipeline_execution_result_left = self.dp[i - num_stages].execution_result.as_ref().unwrap();
                    let new_pipeline_execution_result_right = PipelineExecutionResult::make_base_result(self.merge_stage(virtual_stages, i - num_stages, i - 1, &layers[node_type_idx], node_type_idx as u32));
                    new_dp_state.execution_result = Some(Rc::new(PipelineExecutionResult::new(new_pipeline_execution_result_left, &new_pipeline_execution_result_right)));

                    if self.dp[i].execution_result.is_none() {
                        self.dp[i] = new_dp_state;
                    } else {
                        let new_execution_time = new_dp_state.execution_result.as_ref().unwrap().latency();
                        let old_execution_time = self.dp[i].execution_result.as_ref().unwrap().latency();
                        if new_execution_time < old_execution_time {
                            self.dp[i] = new_dp_state;
                        }
                    }
                }

            }
        }


        if let Some(execution_result) = self.dp[self.dp.len() - 1].execution_result.as_ref() {
            Ok((**execution_result).clone())
        } else {
            Err(PlannerError::new(
                format!("No pipeline template for nodes").as_str(),
            ))?
        }
    }
}