import React, { Fragment, useEffect, useRef, useState } from "react";

import { InferenceSession } from "onnxruntime-web";
import loadImage from "blueimp-load-image";
import { HashLoader } from "react-spinners";
import { createModelCpu, warmupModel } from "./utils";
import { detectObjects } from "./utils/yolov8";

export default function ObjectDetector() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [yolov8, setYolov8] = useState<InferenceSession | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const [nms, setNms] = useState<InferenceSession | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const modelInputShape = [1, 3, 640, 640];

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

  const handleChange = (e: any) => {
    if (!yolov8) return;
    if (!nms) return;
    const url = e.target.files[0];

    loadImage(
      url,
      (img: any) => {
        if ((img as Event).type === "error") {
          return;
        } else {
          if (!canvasRef.current) return;
          const canvas = canvasRef.current;
          const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
          const imageWidth = (img as HTMLImageElement).width;
          const imageHeight = (img as HTMLImageElement).height;
          ctx.drawImage(
            img as HTMLImageElement,
            0,
            0,
            imageWidth,
            imageHeight,
            0,
            0,
            canvas.width,
            canvas.height
          );

          detectObjects(
            canvasRef.current,
            yolov8,
            labels,
            nms,
            { topK: 100, iouThreshold: 0.25, scoreThreshold: 0.45 },
            modelInputShape
          );
        }
      },
      { cover: true, crop: true, canvas: true, crossOrigin: "Anonymous" }
    );
  };

  return (
    <div className="container mx-auto">
      <div className="flex flex-wrap justify-evenly items-center my-5">
        {loading ? (
          <HashLoader color="#36d7b7" />
        ) : (
          <Fragment>
            <div className="">
              <canvas
                id="input-canvas"
                ref={canvasRef}
                width={640}
                height={640}
                className="border border-black"
              />
            </div>
            <div className="">
              <input
                type="file"
                onChange={handleChange}
                className="py-2.5 px-5 mr-2 mb-2 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-200 "
              />
            </div>
          </Fragment>
        )}
      </div>
    </div>
  );
}
