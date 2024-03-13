import fs from "fs";
const segment = require("segment");

const tokenize = (filePath: string) => {
  const data = fs.readFileSync(filePath, "utf-8");
  const arr = data.split("");
  arr.unshift("");
  const charSet = new Set(arr);
  fs.writeFileSync(
    `./src/dataset/charSet.json`,
    JSON.stringify(Array.from(charSet))
  );

  const charMap: Record<string, number> = {};
  Array.from(charSet).forEach((char, index) => {
    charMap[char] = index;
  });
  fs.writeFileSync(`./src/dataset/charMap.json`, JSON.stringify(charMap));

  const charMapReverse: Record<number, string> = {};
  Array.from(charSet).forEach((char, index) => {
    charMapReverse[index] = char;
  });
  fs.writeFileSync(
    `./src/dataset/charMapReverse.json`,
    JSON.stringify(charMapReverse)
  );
};

const str2Token = (str: string, maxLength: number = 100): number[] => {
  const charMap = JSON.parse(
    fs.readFileSync(`./src/dataset/charMap.json`, "utf-8")
  );
  // 补全到最大长度
  const arr = str.split("").map((char) => charMap[char] || 0);
  arr.length = maxLength;
  arr.fill(0, str.length);
  return arr;
};

const token2Str = (tokens: number[]): string => {
  const charMapReverse = JSON.parse(
    fs.readFileSync(`./src/dataset/charMapReverse.json`, "utf-8")
  );
  return tokens.map((token) => charMapReverse[token] || "").join("");
};

const generateTrainingData = (
  filePath: string,
  allowedSegMaxLen: number = 10
) => {
  const data = fs.readFileSync(filePath, "utf-8").replace(/"/g, "");
  const arr = data.split("\r\n").filter((str) => str);

  const seg = new segment();
  seg.useDefault();
  let segizedArr: string[][] = arr.map((str) => {
    return seg.doSegment(str).map((item: { w: string }) => item.w);
  });
  let segStrMaxLen = Math.max(
    ...segizedArr.map((str) => Math.max(...str.map((item) => item.length)))
  );
  if (segStrMaxLen > allowedSegMaxLen) {
    segizedArr = segizedArr.map((strArr) => {
      return strArr.map((str) => {
        if (str.length > allowedSegMaxLen) {
          str = str.slice(0, allowedSegMaxLen);
        }
        return str;
      });
    });
    segStrMaxLen = allowedSegMaxLen;
  }

  const maxSegLen = Math.max(...segizedArr.map((arr) => arr.length));

  const tokenlizedArr = segizedArr.map((strArr, index) => {
    if (strArr.length < maxSegLen) {
      strArr = strArr.concat(Array(maxSegLen - strArr.length).fill(""));
    }
    console.log(`done: ${index + 1}/${segizedArr.length}`);
    return strArr.map((str) => str2Token(str, segStrMaxLen));
  });

  return {
    maxSegLen,
    segStrMaxLen,
    tokenlizedArr,
  };
};

export { tokenize, str2Token, token2Str, generateTrainingData };
