import fs from "fs";
const segment = require("segment");

// 初始化分词器
const seg = new segment();
seg.useDefault();

const doSegment = (sentence: string): string[] => {
  return seg.doSegment(sentence).map((item: { w: string }) => item.w);
};

const tokenlize = (filePath: string, maxWordNum: number = 20) => {
  const data = fs.readFileSync(filePath, "utf-8");
  // 去掉双引号
  const arr = data.split("\r\n").map((str) => str.replace(/"/g, ""));

  // 生成词表
  const wordArr: string[] = [];
  // 生成去重词表
  const wordSet: Set<string> = new Set();
  // 0与空字符串对应
  wordSet.add("");
  arr.forEach((comment, index) => {
    console.log(`(${index + 1}/${arr.length}) handling: ${comment}`);
    doSegment(comment).forEach((word, index) => {
      wordArr.push(word);
      wordSet.add(word);
    });
  });
  fs.writeFileSync(`./src/dataset/wordArr.json`, JSON.stringify(wordArr));
  console.log("wordArr.json generated (1/5)");
  fs.writeFileSync(
    `./src/dataset/wordSet.json`,
    JSON.stringify(Array.from(wordSet))
  );
  console.log("wordSet.json generated (2/5)");

  // 生成词表映射
  const wordMap: Record<string, number> = {};
  Array.from(wordSet).forEach((word, index) => {
    wordMap[word] = index;
  });
  fs.writeFileSync(`./src/dataset/wordMap.json`, JSON.stringify(wordMap));
  console.log("wordMap.json generated (3/5)");

  // 生成词表反向映射
  const wordMapReverse: Record<number, string> = {};
  Array.from(wordSet).forEach((word, index) => {
    wordMapReverse[index] = word;
  });
  fs.writeFileSync(
    `./src/dataset/wordMapReverse.json`,
    JSON.stringify(wordMapReverse)
  );
  console.log("wordMapReverse.json generated (4/5)");

  // 生成句单位数组
  const sentences = arr.map((str, index) => {
    console.log(` (${index + 1}/${arr.length}) handling: ${str}`);
    return sentence2Token(str, maxWordNum);
  });
  fs.writeFileSync(`./src/dataset/sentences.json`, JSON.stringify(sentences));
  console.log("sentences.json generated (5/5)");
};

const sentence2Token = (str: string, maxWordNum: number = 20): number[] => {
  const wordMap = JSON.parse(
    fs.readFileSync(`./src/dataset/wordMap.json`, "utf-8")
  );
  const wordArr = doSegment(str);
  const segArr: number[] = wordArr.map((word: string) => wordMap[word] || 0);
  // 补全到最大长度
  segArr.length = maxWordNum;
  wordArr.length < maxWordNum && segArr.fill(0, wordArr.length);
  return segArr;
};

const token2Sentence = (tokens: number[]): string => {
  const wordMapReverse = JSON.parse(
    fs.readFileSync(`./src/dataset/wordMapReverse.json`, "utf-8")
  );
  return tokens.map((token) => wordMapReverse[token] || "").join("");
};

export { doSegment, tokenlize, sentence2Token, token2Sentence };
