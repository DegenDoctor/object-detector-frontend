import { InferenceSession, Tensor } from "onnxruntime-web";
import cv from "@techstark/opencv-js";

interface YOLOv8Config {
  topK: number; // Integer representing the maximum number of boxes to be selected per class
  iouThreshold: number; // Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
  scoreThreshold: number; // Float representing the threshold for deciding when to remove boxes based on score
}

interface ObjectBox {
  label: number;
  probability: number;
  bounding: number[];
}

/**
 * Preprocessing image
 * @param {HTMLImageElement | HTMLCanvasElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */
export const preprocess = (
  source: HTMLCanvasElement,
  modelWidth: number,
  modelHeight: number
) => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();
  console.log(input.data32F);
  return { input, xRatio, yRatio };
};

/**
 * Detect Image
 * @param {HTMLImageElement | HTMLCanvasElement} source image source
 * @param {InferenceSession} yolov8 YOLOv8 onnxruntime session
 * @param {string[]} labels
 * @param {InferenceSession} nms NMS onnxruntime session
 * @param {YOLOv8Config} config NMS onnxruntime session
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 * @return preprocessed image and configs
 */
export const detectObjects = async (
  source: HTMLCanvasElement,
  yolov8: InferenceSession,
  labels: string[],
  nms: InferenceSession,
  config: YOLOv8Config,
  inputShape: number[]
) => {
  const classLength = labels.length;
  const { input, xRatio, yRatio } = preprocess(source, 640, 640);
  const tensor = new Tensor("float32", input.data32F, inputShape);
  const configTensor = new Tensor(
    "float32",
    new Float32Array([
      classLength,
      config.topK,
      config.iouThreshold,
      config.scoreThreshold,
    ])
  );

  const { output0 } = await yolov8.run({ images: tensor });
  const { selected } = await nms.run({
    detection: output0,
    config: configTensor,
  });

  const boxes = [];

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(
      idx * selected.dims[2],
      (idx + 1) * selected.dims[2]
    ); // get rows
    const box = data.slice(0, 4) as Float32Array;
    const scores = data.slice(4) as Float32Array; // classes probability scores
    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores

    const [x, y, w, h] = [
      (box[0] - 0.5 * box[2]) * xRatio, // upscale left
      (box[1] - 0.5 * box[3]) * yRatio, // upscale top
      box[2] * xRatio, // upscale width
      box[3] * yRatio, // upscale height
    ]; // keep boxes in maxSize range

    boxes.push({
      label: label,
      probability: score,
      bounding: [x, y, w, h], // upscale box
    }); // update boxes to draw later
  }

  console.log(boxes);
  renderBoxes(source, boxes, labels);
  return;
};

export const renderBoxes = (
  canvas: HTMLCanvasElement,
  boxes: ObjectBox[],
  labels: string[]
) => {
  const ctx = canvas.getContext("2d")!;
  // ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  // font configs
  const font = "20px Arial";
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const klass = labels[box.label];
    const color = "#000";
    const score = (box.probability * 100).toFixed(1);
    const [x1, y1, width, height] = box.bounding;

    // draw box.
    // ctx.fillStyle = Colors.hexToRgba(color, 0.2);
    // ctx.fillRect(x1, y1, width, height);
    // draw border box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, width, height);

    // draw the label background.
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(klass + " - " + score + "%").width;
    const textHeight = parseInt(font, 10); // base 10
    const yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    );

    // Draw labels
    ctx.fillStyle = "#ffffff";
    ctx.fillText(
      klass + " - " + score + "%",
      x1 - 1,
      yText < 0 ? 1 : yText + 1
    );
  });
};
