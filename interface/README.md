# DSP-Audio-Collector
Web app created to collect audios for course project

Recorder使用说明

1. 地址在[https://ichn-hu.github.io/DSP-Audio-Collector/](https://ichn-hu.github.io/DSP-Audio-Collector/)，请用chrome or firefox or edge打开，目前看来仿佛各个浏览器录音效果有区别，需要进一步研究
2. 提示请求使用麦克风，请确认授予权限
3. 输入学号，第二个输入框内的值为当前录音条目编号，这个值可以帮助暂停/恢复录制
4. 一个条目完整的录音过程如下
    1. 点击开始录音
    2. 2s 时间念出提示词
    3. 播放录音，如果觉得不行，可以回到第1步
    4. 点击下载，完成
5. 总共400个条目，出于抗疲劳以及给数据增加variance的考虑，录音顺序是打乱的，但这个打乱的顺序是不会变的，可以通过记住当前录制了多少条语音保存工作进度，下次恢复时+1输入到第二个输入框即可恢复工作╰(￣ω￣ｏ)
