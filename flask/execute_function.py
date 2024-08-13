import subprocess

def execute_colmap_in_container(folder_name):
    container_name = "online-cgc-colmap-1"  # Name of the COLMAP container
    path_to_colmap_binaries = "/usr/local/bin"  # Update this path as necessary
    images_folder = f"/app/data/{folder_name}/images"
    output_folder = f"/app/data/{folder_name}"
    command_to_execute = f"/app/colmap/colmap.sh {path_to_colmap_binaries} {images_folder} {output_folder}"
    docker_exec_command = f"docker exec -i {container_name} {command_to_execute}"
    #docker_exec_command = "docker exec -i online-cgc-colmap-1 /app/colmap/colmap.sh /usr/local/bin /app/data/2024-05-13_13-03-29/images /app/data/2024-05-13_13-03-29/output" 

    try:
        result = subprocess.run(docker_exec_command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with return code {e.returncode}:\n{e.stderr}\n{e.stdout}"
    

def execute_sam_in_container(folder_name):
    container_name = "online-cgc-sam-1"  # Name of the SAM container
    sam_checkpoint = f"/app/sam/sam_vit_h_4b8939.pth"  # Update this path as necessary
    data_dir = f"/app/data/{folder_name}"
    command_to_execute = f"/app/sam/auto_sam.sh {sam_checkpoint} {data_dir}"
    docker_exec_command = f"docker exec -i {container_name} {command_to_execute}"

    try:
        result = subprocess.run(docker_exec_command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with return code {e.returncode}:\n{e.stderr}\n{e.stdout}"
    

def execute_cgc_in_container(folder_name):
    container_name = "online-cgc-cgc-1"  # Name of the CGC container
    cgc_config = f"/app/cgc/config/gaussian_dataset/train.json"  # Update this path as necessary
    data_dir = f"/app/data/{folder_name}"
    command_to_execute = f"/app/cgc/train.sh {data_dir} {cgc_config}"
    docker_exec_command = f"docker exec -i {container_name} {command_to_execute}"

    try:
        result = subprocess.run(docker_exec_command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with return code {e.returncode}:\n{e.stderr}\n{e.stdout}"

# Example call to execute_colmap_in_container
if __name__ == "__main__":
    folder_name = "example_folder"  # Replace with actual folder name
    print(execute_colmap_in_container(folder_name))




