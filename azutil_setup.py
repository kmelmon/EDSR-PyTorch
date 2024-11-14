# Description: This script sets up an Azure environment for training a PyTorch model.
# It uses the Azure Machine Learning SDK to create a compute target, environment, and experiment.
# The script then submits a training job to the compute target.


from azureml.core import Workspace, ComputeTarget, Environment, ScriptRunConfig, Experiment
from azureml.core.runconfig import MpiConfiguration, DockerConfiguration

import setup_EDSR_args

setup_EDSR_args.parser.add_argument('--cluster', type=str, help='cluster to use', default='cpu-cluster-large')
setup_EDSR_args.parser.add_argument('--data_store_training_data', type=str, help='data_store', default='trainingdata_sigma')
setup_EDSR_args.parser.add_argument('--shared_mem_size', type=int, help='shared memory size', default=640)
setup_EDSR_args.parser.add_argument('--mount_training_data', type=str, help='mount directory name in Azure Storage for training data', default='')
setup_EDSR_args.parser.add_argument('--experiment_name', type=str, help='experiment name', default='EDSR')

keys_not_included = {
    'cluster' : None,
    'data_store_training_data' : None,
    'env_file' : None,
    'shared_mem_size' : None,
    'mount_training_data' : None,
    'experiment_name' : None,
}

def setup_azure():

    args = setup_EDSR_args.parser.parse_args()

    # Set up MPI configuration
    #distr_config = MpiConfiguration(process_count_per_node=1, node_count=1)

    # Set up Docker configuration
    docker_base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04'
    pytorch_env = Environment.from_conda_specification(name = "RDNN-GPU", file_path = './conda_dependencies.yml')
    pytorch_env.docker.base_image = docker_base_image

    docker_config = DockerConfiguration(use_docker=True, shm_size=f"{args.shared_mem_size}g")

    # Set up Azure environment
    az_ws = Workspace.get(name = "GfxMLTrainingGPUWorkspace1",  subscription_id = "68d80131-d556-4763-8084-2a66f90a8efd", resource_group= "GfxMLTraining")
    az_ds_training_data = az_ws.datastores[args.data_store_training_data]
    az_ds_ref_rootfolder_training_data = az_ds_training_data.path(args.mount_training_data).as_mount()
    az_target = ComputeTarget(az_ws, args.cluster)

    args_dict = vars(args)
    arguments_list = []
    for key, value in args_dict.items() :

        if key in keys_not_included :
            continue

        arguments_list.append(f'--{key}={value}')

    arguments_list.append(f'--root_folder_training_data={str(az_ds_ref_rootfolder_training_data)}')
    
    print(f"argument list:{arguments_list}")

    az_config = ScriptRunConfig(
        source_directory=".",
        script="setupEDSR.py",
        compute_target=az_target,
        docker_runtime_config=docker_config,
        environment=pytorch_env,
        arguments=arguments_list,
        #distributed_job_config=distr_config
    )
    az_config.run_config.data_references[az_ds_ref_rootfolder_training_data.data_reference_name] = az_ds_ref_rootfolder_training_data.to_config()

    # Setup Azure experiment
    az_exp = Experiment(az_ws, args.experiment_name)
    run = az_exp.submit(az_config)
    print(run)
    run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    setup_azure()