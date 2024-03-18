import fs from "fs";
const segment = require("segment");

// 初始化分词器
const seg = new segment();
seg.useDefault();

const doSegment = (sentence: string): string[] => {
  return seg.doSegment(sentence).map((item: { w: string }) => item.w);
};

const tokenlize = (filePath: string) => {
  const data = fs.readFileSync(filePath, "utf-8");
  // 去掉双引号
  const arr = data.split("\r\n").map((str) => str.replace(/"/g, ""));

  // 生成词表
  fs.writeFileSync(
    `./src/dataset/wordArr.json`,
    JSON.stringify(doSegment(arr.join("")))
  );

  // 生成唯一词表
  const wordSet: Set<string> = new Set();
  // 0与空字符串对应
  wordSet.add("");
  arr.forEach((comment) => {
    doSegment(comment).forEach((word) => wordSet.add(word));
  });

  fs.writeFileSync(
    `./src/dataset/wordSet.json`,
    JSON.stringify(Array.from(wordSet))
  );

  // 生成词表映射
  const wordMap: Record<string, number> = {};
  Array.from(wordSet).forEach((word, index) => {
    wordMap[word] = index;
  });
  fs.writeFileSync(`./src/dataset/wordMap.json`, JSON.stringify(wordMap));

  // 生成词表反向映射
  const wordMapReverse: Record<number, string> = {};
  Array.from(wordSet).forEach((word, index) => {
    wordMapReverse[index] = word;
  });
  fs.writeFileSync(
    `./src/dataset/wordMapReverse.json`,
    JSON.stringify(wordMapReverse)
  );
};

const sentence2Token = (str: string, segMaxLen: number = 10): number[] => {
  const wordMap = JSON.parse(
    fs.readFileSync(`./src/dataset/wordMap.json`, "utf-8")
  );
  const wordArr = doSegment(str);
  const segArr: number[] = wordArr.map((word: string) => wordMap[word] || 0);
  // 补全到最大长度
  segArr.length = segMaxLen;
  wordArr.length < segMaxLen && segArr.fill(0, wordArr.length);
  return segArr;
};

const token2Sentence = (tokens: number[]): string => {
  const wordMapReverse = JSON.parse(
    fs.readFileSync(`./src/dataset/wordMapReverse.json`, "utf-8")
  );
  return tokens.map((token) => wordMapReverse[token] || "").join("");
};

const generateTrainingData = (
  xsfilePath: string,
  ysfilePath: string,
  SegMaxLen: number = 10
) => {
  const xsdata = fs.readFileSync(xsfilePath, "utf-8").replace(/"/g, "");
  const sentenceArr = xsdata.split("\r\n").filter((str) => str);

  const ysdata = fs.readFileSync(ysfilePath, "utf-8").replace(/"/g, "");
  const rateArr = ysdata
    .split("\r\n")
    .filter((str) => str)
    .map(Number);

  if (sentenceArr.length !== rateArr.length) {
    throw new Error("xs 与 ys 长度不一致，请检查");
  }

  const sentenceTokenlizedArr = sentenceArr.map((str) => {
    return sentence2Token(str, SegMaxLen);
  });

  return {
    xs: sentenceTokenlizedArr,
    ys: rateArr,
  };
};

export { doSegment, tokenlize, sentence2Token, token2Sentence, generateTrainingData };
