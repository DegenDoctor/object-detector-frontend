import { useEffect, useState } from "react";
import { InferenceSession } from "onnxruntime-web";
import { createModelCpu, warmupModel } from "../utils";

const useYoloEngine = () => {
  const [yolov8, setYolov8] = useState<InferenceSession | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const [nms, setNms] = useState<InferenceSession | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      setLoading(true);
      const [yolov8, labels, nms] = await Promise.all([
        createModelCpu(`/yolov8n.onnx`),
        fetch("/labels.json").then<string[]>((res) => res.json()),
        createModelCpu(`/nms.onnx`),
      ]);

      await warmupModel(yolov8, [1, 3, 640, 640]);
      setYolov8(yolov8);
      setLabels(labels);
      setNms(nms);
      setLoading(false);
    })();
  }, []);

  return {
    yolov8,
    labels,
    nms,
    loading,
  };
};

export default useYoloEngine;
