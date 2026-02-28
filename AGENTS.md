我的计划是教会图像编辑模型训练mel 频谱的修改。
mel 频谱遵照一个 neural vocoder 的定义，所以可以转为 wav，从而相当于实现让一个图像编辑模型能做 wav->wav 的修改。
计划先收集一万个句子（量够吗？）
target mel 则是由一个 serious TTS（如 qwen tts）来生成 wav 然后转 mel，来生成。
source mel 计划由 espeak 由这些句子生成 wav，然后转 mel，然后有一半概率和 target mel 之间做 DP 来 DTW 对齐，来生成。
