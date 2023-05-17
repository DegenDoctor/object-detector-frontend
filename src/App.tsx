import React, { useEffect, useRef, useState } from "react";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import { Tensor, InferenceSession } from "onnxruntime-web";
import loadImage from "blueimp-load-image";
import { createModelCpu, runModel } from "./utils";

export default function ObjectDetector() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [sessionRunning, setSessionRunning] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      const session = await createModelCpu(`/yolov8n.onnx`);
      console.log(session);
      setSession(session);
    })();
  }, []);

  const preprocess = (ctx: CanvasRenderingContext2D): Tensor => {
    const imageData = ctx.getImageData(
      0,
      0,
      ctx.canvas.width,
      ctx.canvas.height
    );
    const { data, width, height } = imageData;
    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    ops.assign(
      dataProcessedTensor.pick(0, 0, null, null),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 1, null, null),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 2, null, null),
      dataTensor.pick(null, null, 2)
    );

    const tensor = new Tensor("float32", new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  };

  const postprocess = async (tensor: Tensor, inferenceTime: number) => {
    console.log("outputTensor", tensor);
    console.log("inferenceTime", inferenceTime);

    try {
      const originalOutput = new Tensor(
        "float32",
        tensor.data as Float32Array,
        [1, 125, 13, 13]
      );

      console.log(originalOutput);
    } catch (e) {
      alert("Model is not valid!");
    }
  };

  const runSession = async (ctx: CanvasRenderingContext2D) => {
    if (!session) return;
    setSessionRunning(true);
    const data = preprocess(ctx);
    const [outputTensor, inferenceTime] = await runModel(session, data);

    postprocess(outputTensor, inferenceTime);
  };

  const handleChange = (e: any) => {
    const url = e.target.files[0];
    console.log(url);

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

          runSession(ctx);
        }
      },
      { cover: true, crop: true, canvas: true, crossOrigin: "Anonymous" }
    );
  };

  return (
    <div className="container mx-auto">
      <div className="flex flex-wrap justify-evenly items-center my-5">
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
      </div>
    </div>
  );
}
