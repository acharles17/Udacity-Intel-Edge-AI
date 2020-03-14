alias "Model = /opt/intel/openvino//deployment_tools/model_optimizer"
alias "


python3 mo_tf.py --input_model  ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --reverse_input_channels --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config

python3 /opt/intel/openvino//deployment_tools/model_optimizer/mo_tf.py --input_model=ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels
/opt/intel/openvino//deployment_tools/model_optimizer
