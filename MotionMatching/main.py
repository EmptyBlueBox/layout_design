from MotionMatching.motion_matching import MotionMatching
from MotionMatching.visualization import visualize_motion
import yaml
import os

def main(keypoints=None, threshold=0.5):
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        config["database"]["AMASS_path"] = os.path.join(os.path.dirname(__file__), config["database"]["AMASS_path"])
        config["database"]["AMASS_cache_path"] = os.path.join(os.path.dirname(__file__), config["database"]["AMASS_cache_path"])
    
    if keypoints is not None:
        config["motion_matching"]["keypoint_list"] = keypoints
    config["motion_matching"]["touch_threshold"] = threshold
    
    motion_matching = MotionMatching(config)
    motion = motion_matching.run()
    # visualize_motion(motion, config)
    return motion
    
if __name__ == "__main__":
    main()