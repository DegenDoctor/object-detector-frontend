import React, { useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { HashLoader } from "react-spinners";

import { detectObjects } from "./utils/yolov8";
import useYoloEngine from "./hooks/useYoloEngine";

export default function ObjectDetector() {
  const [files, setFiles] = useState<any[]>([]);
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

    detectObjects(
      imageRef.current,
      canvasRef.current,
      yolov8,
      labels,
      nms,
      100,
      0.25,
      0.5
    );
  }, [files, imageRef, canvasRef, loading, labels, nms, yolov8]);

  return (
    <div className="container mx-auto">
      {loading ? (
        <HashLoader color="#36d7b7" />
      ) : (
        <div className="bg-white p-5 rounded-lg my-5">
          <h1 className="text-lg">YOLOv8 Model</h1>

          <div className="flex">
            <div className="flex-1">
              <div {...getRootProps({ className: "dropzone" })}>
                <input {...getInputProps()} />
                <p>Drag 'n' drop some files here, or click to select files</p>

                {files.length > 0 && (
                  <div className="relative">
                    <img
                      ref={imageRef}
                      src={files[0].preview}
                      alt="img-preview"
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute top-0 left-0 w-full h-full"
                    />
                  </div>
                )}
              </div>
            </div>
            <div className="flex-1">
              <div className="">settings</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
