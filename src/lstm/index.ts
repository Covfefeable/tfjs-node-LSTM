import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import { generateTrainingData, tokenlize } from "../utils";
const segment = require("segment");

const config = {
  modelSavePath: "file://./src/model",
  xsDataSetPath: "./src/dataset/comments.txt",
  ysDataSetPath: "./src/dataset/rate.txt",
};

const train = async () => {
  tokenlize(config.xsDataSetPath);
  console.log('tokenlized')
  const { xs, ys } = generateTrainingData(
    config.xsDataSetPath,
    config.ysDataSetPath,
    20
  );

  console.log('training data generated')

  const input = tf.tensor2d(xs);
  const output = tf.oneHot(tf.tensor(ys).cast("int32"), 2);

  console.log(input, output);

  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [20],
      units: 256,
      activation: "relu",
    })
  );
  // todo
  model.add(
    tf.layers.dense({
      units: 2,
      activation: "sigmoid",
    })
  );
  model.compile({
    loss: "binaryCrossentropy",
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
};

export { train };
