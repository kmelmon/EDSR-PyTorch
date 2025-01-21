from azureml.core import Workspace, ComputeTarget, Environment, ScriptRunConfig, Experiment
from azureml.core.runconfig import MpiConfiguration, DockerConfiguration

def setup_azure():

    #distr_config = MpiConfiguration(process_count_per_node=4, node_count=1)

    docker_base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04'
    pytorch_env = Environment.from_conda_specification(name = "RDNN-GPU", file_path = './conda_dependencies.yml')
    pytorch_env.docker.base_image = docker_base_image

    shm_size = 80 * max(4, 1)
    docker_config = DockerConfiguration(use_docker=True, shm_size=f"{shm_size}g")

    az_ws = Workspace.get(name = "GfxMLTrainingGPUWorkspace1",  subscription_id = "68d80131-d556-4763-8084-2a66f90a8efd", resource_group= "GfxMLTraining")
    az_ds = az_ws.datastores["trainingdata_sigma"]
    az_ds_ref_div2k = az_ds.path("Data/PC_Synthetic/AgeOfEmpires/FXAA").as_mount()

    #for testing
    args = ["--client_id", "7ea1e259-3558-425a-a8b9-106b5b76ac20", 
            "--container_name", "trainingblob", 
            "--save_inference_to_azure", "Data/PC_Synthetic/AgeOfEmpires/FXAA/1280x800SyntheticVGG", 
            "--azureml", 
            "--dir_data", str(az_ds_ref_div2k), 
            "--pre_train", "models/model_best_VGG_FXAA.pt", 
            "--dir_demo", "1280x800PreScaledBilinear/1280x800", 
            "--test_only", "--save_results", "--data_test", "Demo", "--model", "EDSR", "--downscale", "--scale", "1", "--patch_size", "96", "--n_resblocks", "32", "--n_feats", "128", "--res_scale", "0.1"]

    az_target = ComputeTarget(az_ws, "gpu-4xv100-sc-1")
    az_config = ScriptRunConfig(
        source_directory="src",
        script="main.py",
        compute_target=az_target,
        # environment=az_env,
        docker_runtime_config=docker_config,
        environment=pytorch_env,
        arguments=args,
        # arguments=[str(az_ds_ref)],
        #distributed_job_config=distr_config
    )
    az_config.run_config.data_references[az_ds_ref_div2k.data_reference_name] = az_ds_ref_div2k.to_config()

    # az_config.run_config.data_references[az_ds_ref.data_reference_name] = az_ds_ref.to_config()
    az_exp = Experiment(az_ws, "EDSR")
    run = az_exp.submit(az_config)
    print(run)
    run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    setup_azure()