# EEG_vision_imagination

## Context

**SEMI-Can we read our minds with EEG signals?**

**Context** : recent research projects are dedicated to robotics using our mind and EEG devices. However, some are simply based on detecting eye movements. What about thinking images or actions and recognizing them? This work will study the techniques used to identify objects (for example, seeing or imagining an apple) and actions (raising the arm or closing the hand) using EEG devices.

**Objectives** : study algorithms and techniques based on EEG signals used to recognize objects or actions. Compare them and build a prototype to recognize a basic set of concrete objects (an apple, a table, a hammer, a pencil, a hand). Do the same for actions (eye, arm, hand or leg movements). Analyze the signals looking for patterns in each case. Evaluate the results.

## Experiment

Our experiment adopts the method of the OpenBCI official (cat and dog) experiment, generates a video for the experimenter (me) to watch, and uses the device to collect the brain EEG signal while watching the video and store it in a txt file.

Specifically, we used American Sign Language letters and pictures of the alphabet to generate videos. The experimenter will observe these letters and imagine the letters just observed after observing them, so our signals are also called visual signals/imagination signals, sign language signal/alphabet signal.

Our task is to perform multi-classification and binary classification tasks on these signals.

## Feature extraction

After obtaining the EEG signals, we compared the EEG signals in different ways, and also extracted the EEG features. Finally, 86 features can be extracted for each channel. These features not only help us in the comparison process, but also in the machine learning process played a big role.

## Models

We have tested a lot of machine learning and deep learning models and neural networks, and the best performance is the random forest model, with an accuracy rate of 90-98%. (Please refer to the notebook folder/internship report for more details)

## Future work

This will/could be a step forward for brain typing.

## Contact

Due to the limitation of github, the data and video are not fully uploaded. If you need, or if you encounter any problems in executing the code, please contact me.

- gmail: xulintaofr@gmail.com/lintao.xu@edu.esiee.fr
- github link:https://github.com/xul-ops/EEG_vision_imagination-
