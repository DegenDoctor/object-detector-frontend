import { useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import Slider from "rc-slider";
import { FaUpload } from "react-icons/fa";
import { HashLoader } from "react-spinners";

import { detectObjects } from "./utils/yolov8";
import useYoloEngine from "./hooks/useYoloEngine";
import "rc-slider/assets/index.css";

export default function ObjectDetector() {
  const [files, setFiles] = useState<any[]>([]);
  const [scoreThreshold, setScoreThreshold] = useState(0.45);
  const [iouThreshold, setIouThreshold] = useState(0.25);
  const [imgLoaded, setImgLoaded] = useState<boolean>(false);
  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      "image/*": [],
    },
    onDrop: (acceptedFiles) => {
      setFiles(
        acceptedFiles.map((file) =>
          Object.assign(file, {
            preview: URL.createObjectURL(file),
          })
        )
      );
    },
    multiple: false,
  });
  const { yolov8, labels, nms, loading } = useYoloEngine();
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (
      files.length === 0 ||
      !imageRef.current ||
      !canvasRef.current ||
      loading ||
      !yolov8 ||
      !nms
    )
      return;

    if (!imgLoaded) return;

    console.log("detecting");
    detectObjects(
      imageRef.current,
      canvasRef.current,
      yolov8,
      labels,
      nms,
      100,
      iouThreshold,
      scoreThreshold
    );
  }, [
    files,
    imageRef,
    canvasRef,
    loading,
    labels,
    nms,
    yolov8,
    imgLoaded,
    scoreThreshold,
    iouThreshold,
  ]);

  return (
    <div className="container mx-auto">
      {loading ? (
        <div className="flex w-full h-[100vh] items-center justify-center">
          <div>
            <HashLoader color="#36d7b7" />
            <h1 className="text-sm text-[#36d7b7] mt-3">loading...</h1>
          </div>
        </div>
      ) : (
        <div className="bg-white p-5 rounded-lg my-5">
          <h1 className="text-lg mb-5 font-bold">YOLOv8 Model</h1>

          <div className="flex gap-5">
            <div className="w-full">
              <div className="">Settings</div>

              <div className="flex gap-5">
                <h3>Image Size</h3>
                <h3>640px</h3>
              </div>
              <div className="flex gap-5 items-center">
                <h3 className="w-1/3">Confidence Threshold</h3>
                <div className="flex items-center gap-3 flex-1">
                  <Slider
                    min={0}
                    max={1}
                    step={0.01}
                    value={scoreThreshold}
                    onChange={(v) => setScoreThreshold(v as number)}
                  />
                  <span className="w-8">{scoreThreshold}</span>
                </div>
              </div>
              <div className="flex gap-5 items-center">
                <h3 className="w-1/3">IoU Threshold</h3>
                <div className="flex items-center gap-3 flex-1">
                  <Slider
                    min={0}
                    max={1}
                    step={0.01}
                    value={iouThreshold}
                    onChange={(v) => setIouThreshold(v as number)}
                  />
                  <span className="w-8">{iouThreshold}</span>
                </div>
              </div>
            </div>
            <div className="w-full">
              <div {...getRootProps({ className: "dropzone" })}>
                <input {...getInputProps()} />
                {files.length === 0 && (
                  <div className="text-center border border-dashed p-3">
                    <div className="flex justify-center">
                      <FaUpload size={30} />
                    </div>
                    Drag and drop your image here or browse your computer
                  </div>
                )}

                {files.length > 0 && (
                  <div className="relative flex items-center">
                    <img
                      ref={imageRef}
                      src={files[0].preview}
                      onLoad={() => setImgLoaded(true)}
                      alt="img-preview"
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute top-0 left-0 w-full h-full"
                      width={640}
                      height={640}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
