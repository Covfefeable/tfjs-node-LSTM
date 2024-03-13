import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import { generateTrainingData, str2Token, token2Str, tokenize } from "../utils";
const segment = require("segment");

const config = {
  modelSavePath: "file://./src/model",
  dataSetPath: "./src/dataset/comments.txt",
};

const train = async () => {
  tokenize(config.dataSetPath);
  const data = generateTrainingData(config.dataSetPath);
  const rateArr = fs
    .readFileSync("./src/dataset/rate.txt", "utf-8")
    .split("\r\n")
    .map((item: string) => Number(item));

  const input = tf.tensor3d(data.tokenlizedArr);
  const output = tf.oneHot(tf.tensor(rateArr).cast("int32"), 2);

  const model = tf.sequential();

  model.add(
    tf.layers.lstm({
      units: 256,
      inputShape: [data.maxSegLen, data.segStrMaxLen],
      returnSequences: true,
    })
  );
  model.add(
    tf.layers.lstm({
      units: 128,
      returnSequences: true,
    })
  );
  model.add(
    tf.layers.lstm({
      units: 64,
    })
  );
  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax",
    })
  );

  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.1),
    metrics: ["accuracy"],
  });
  model.summary();

  await model.fit(input, output, {
    epochs: 10,
    shuffle: true,
    // @ts-ignore
    onEpochEnd: (epoch, logs) => {
      console.log("Epoch: " + epoch + " Loss: " + logs.loss);
    },
  });

  await model.save(config.modelSavePath);

  const result = await predict("测试");
  console.log(result);
};

const predict = async (str: string) => {
  const model = await tf.loadLayersModel(config.modelSavePath + "/model.json");
  const seg = new segment();
  seg.useDefault();
  const segArr: string[] = seg
    .doSegment(str)
    .map((item: { w: string }) => item.w);
  let tokenlizedArr = segArr.map((str) => str2Token(str, 10));
  if (tokenlizedArr.length < 36) {
    tokenlizedArr = tokenlizedArr.concat(
      Array(36 - tokenlizedArr.length).fill(str2Token("", 10))
    );
  }
  console.log(tokenlizedArr);
  const input = tf.tensor3d([tokenlizedArr]);
  const result = model.predict(input);
  // @ts-ignore
  const rate = result.dataSync()[0];
  return rate;
};

export { train };
