from perception_eval.tool import PerceptionAnalyzer3D
from perception_eval.config import PerceptionEvaluationConfig
import yaml
from pathlib import Path
import os
import matplotlib.pyplot as plt

# basic settings
data_dir = "eval_test/"
scenario_path = "/home/yoshiri/extension_ws/src/tracking_validation/tracking_validation/config/scenario.ja.yaml"
result_archive_path = Path("eval_test")
dataset_path = ""


perception_eval_log_path = result_archive_path.parent.joinpath(
            "perception_eval_log"
        ).as_posix()


# load config
with open(scenario_path, "r") as scenario_file:
    scenario_yaml_obj = yaml.safe_load(scenario_file)
p_cfg = scenario_yaml_obj["Evaluation"]["PerceptionEvaluationConfig"]

evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
                dataset_paths=dataset_path,
                frame_id="map",
                merge_similar_labels=False,
                result_root_directory=os.path.join(
                    perception_eval_log_path, "result", "{TIME}"
                ),
                evaluation_config_dict=p_cfg["evaluation_config_dict"],
                load_raw_data=False,
            )


# load pkl
analyzer = PerceptionAnalyzer3D(evaluation_config)
result_pkl = data_dir + "scene_result.pkl"


analyzer.add_from_pkl(result_pkl)
analyzer.plot_error(["x","y","yaw"])
analyzer.box_plot(["x","y","yaw"])
print(analyzer.summarize_score()) # pandas でスコアが出る

final_score_df = analyzer.summarize_score()
# bar plot

for col in final_score_df.columns:
    # create subfigure with new figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    final_score_df[col].plot.bar(ax=ax)
    ax.set_title(col)

plt.show()