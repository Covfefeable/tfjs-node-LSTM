import * as tf from "@tensorflow/tfjs-node";
import { sentence2Token, tokenlize } from "../utils";
import path from "path";
import fs, { readFileSync } from "fs";

const config = {
  maxWordNum: 20,
  dataSetDir: path.join(path.resolve("./"), "./src/dataset"),
  modelDir: path.join(path.resolve("./"), "./src/model/emotion"),
};

const emotionTrain = async () => {
  if (!fs.existsSync(`${config.dataSetDir}/wordArr.json`)) {
    console.log("preparing training data, it may take a while");
    tokenlize(`${config.dataSetDir}/comments.txt`);
  }

  const wordSet = JSON.parse(
    fs.readFileSync(`${config.dataSetDir}/wordSet.json`, "utf-8")
  );

  const sentences: number[][] = JSON.parse(
    fs.readFileSync(`${config.dataSetDir}/sentences.json`, "utf-8")
  );
  const rateText = readFileSync(`${config.dataSetDir}/rate.txt`, "utf-8");
  const rate: number[] = rateText
    .split("\r\n")
    .map((str) => str.replace(/"/g, ""))
    .map(Number);
  const input = tf.tensor2d(sentences);
  const output = tf.oneHot(tf.tensor(rate).cast("int32"), 2);

  const model = tf.sequential();
  model.add(
    tf.layers.embedding({
      inputDim: wordSet.length,
      outputDim: 8,
      inputLength: config.maxWordNum,
    })
  );

  model.add(
    tf.layers.lstm({
      units: 24,
      returnSequences: true,
    })
  );

  model.add(
    tf.layers.dropout({
      rate: 0.1,
    })
  );

  model.add(
    tf.layers.lstm({
      units: 24,
      returnSequences: false,
    })
  );
  model.add(tf.layers.dense({ units: 2, activation: "softmax" }));
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.002),
    metrics: ["accuracy"],
  });
  model.summary();
  await model.fit(input, output, {
    epochs: 50,
    shuffle: true,
    callbacks: {
      onEpochEnd: async () => {
        await model.save(`file://${config.modelDir}`);
        await testAccuracy();
      },
    },
  });
};

const emotionPredict = async (input: string) => {
  const model = await tf.loadLayersModel(
    `file://${config.modelDir}/model.json`
  );
  const tokenizedInput = sentence2Token(input, config.maxWordNum);
  const inputTensor = tf.tensor2d([tokenizedInput]);
  const result = await model.predict(inputTensor);
  console.log(
    (result as tf.Tensor).dataSync(),
    (result as tf.Tensor).argMax(1).dataSync()[0]
  );
  return (result as tf.Tensor).argMax(1).dataSync()[0];
};

const testAccuracy = async () => {
  let currectCount = 0;
  let wrongCount = 0;
  const model = await tf.loadLayersModel(
    `file://${config.modelDir}/model.json`
  );
  const rateText = readFileSync(
    `${config.dataSetDir}/test-rate-300.txt`,
    "utf-8"
  );
  const currectAnswer: number[] = rateText
    .split("\r\n")
    .map((str) => str.replace(/"/g, ""))
    .map(Number);
  const commentsText = readFileSync(
    `${config.dataSetDir}/test-comments-300.txt`,
    "utf-8"
  );
  const comments: number[][] = commentsText
    .split("\r\n")
    .map((str) => str.replace(/"/g, ""))
    .map((str) => sentence2Token(str, config.maxWordNum));

  comments.forEach(async (comment, index) => {
    const inputTensor = tf.tensor2d([comment]);
    const result = await model.predict(inputTensor);
    const predict = (result as tf.Tensor).argMax(1).dataSync()[0];
    if (predict === currectAnswer[index]) {
      currectCount++;
    } else {
      wrongCount++;
    }
    index === 299 &&
      console.log(
        `currectCount: ${currectCount}, wrongCount: ${wrongCount}， accuracy: ${
          currectCount / (currectCount + wrongCount)
        }`
      );
  });
};

export { emotionTrain, emotionPredict };
