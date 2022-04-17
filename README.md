# ML4ES2-2021-2022
Machine learning for Earth Sciences F2021-S2022 by [Mikhail Krinitskiy](https://sail.ocean.ru/viewuser.php?user=krinitsky) ([RG link](https://www.researchgate.net/profile/Mikhail_Krinitskiy))

Машинное обучение в науках о Земле. Читает [Михаил Криницкий](https://sail.ocean.ru/viewuser.php?user=krinitsky)<br />

[Метод оценки домашних заданий.](./homeworks_policy.md)<br />

[Правила распространения источников](./resources_policy.md)<br />


-------

| Title | Date | Topic | Content |
| ----- | ---- | ----- | ------- |
|       |      | I семестр |    |
| Лекция 2 | 07.10.2021 | Метод вычисления градиентов функции потерь.<br />Технические аспекты реализации backpropagation | [видеозапись](https://ml4es.ru/links/ML4ES2-2021-10-07.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect02) |
| ДЗ №1 | 07.10.2021<br />deadline: **19.10.2021** | Реализация однослойной линейной нейросети на numpy | [Описание ДЗ и данные](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/HW01)<br />[LEADERBOARD](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/blob/main/HW01/README.md) |
| Лекция 3 | 12.10.2021 | Оптимизация нейросетей. Свойства ландшафта функции потерь.<br />Методы оптимизации. Градиентные методы: GD, SGD. | [материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect03) |
| Лекция 4 | 19.10.2021 | Градиентные методы оптимизации нейросетей.<br />SGD+momentum, Nesterov momentum, RMSProp/AdaGrad, Adam | [видеозапись](https://ml4es.ru/links/ML4ES2-2021-10-19.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect04) |
| Лекция 5 | 02.11.2021 | Градиентная оптимизация нейросетей: расписание шага обучения | [видеозапись](https://ml4es.ru/links/ML4ES2-2021-11-02.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect05) |
| Лекция 6 | 09.11.2021 | Дисперсия карт активаций<br />Роль начального приближения в обучении нейросетей | [видеозапись](https://ml4es.ru/links/ML4ES2-2021-11-09.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect06) |
| Лекция 7 | 16.11.2021 | Дисперсия карт активаций для глубоких нейросетей<br />Варианты инициализации весов, улучшающих сходимость обучения. | [видеозапись](https://ml4es.ru/links/ML4ES2-2021-11-16-Lect07.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect07) |
| ДЗ №2 | 16.11.2021<br />deadline: **30.11.2021** | Реализация многослойной нейросети с нелинейными функциями активации на numpy | [Описание ДЗ и данные](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/HW02) |
| Лекция 8 | 23.11.2021 | Дисперсия карт активаций для глубоких нейросетей<br />Нормализация распределений активаций. Пакетная нормализация (Batch normalization)<br />Pytorch как библиотека для программирования искусственных нейронных сетей. | [видеозапись](https://ml4es.ru/links/ML4ES2-2021-11-23-Lect08.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect08) |
| Лекция 9 | 07.12.2021 | Регуляризации и искусственное дополнение данных (data augmentation) | [видеозапись](https://ml4es.ru/links/ML4ES2-2021-12-07-Lect09.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect09) |
| Лекция 10 | 14.12.2021 | Семинар: обучение ИНС с использованием фреймворка Pytorch.<br />Мониторинг процесса обучения. | [видеозапись](https://ml4es.ru/links/ML4ES2-2021-12-14-Lect10.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect10) |
| Лекция 11 | 08.02.2022 | Инициализация нейросетей, пакетная нормализация (повторение)<br />Функции активации. | [видеозапись](https://ml4es.ru/links/ML4ES2-2022-02-08-Lect11.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect11) |
| Лекция 12 | 22.02.2022 | Свёрточные нейронные сети: предпосылки и обоснование введения свёрточной операции. Циркулянт.<br />(по материалам лекций 2021 г.) | [Видеозапись: функции активации](https://ml4es.ru/links/MailRU-NN-Lect06-AF.mp4)<br />[Видеозапись: свёрточная операция](https://ml4es.ru/links/MailRU-NN-Lect06-Conv.mp4)<br />[Материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect12) |
| Лекция 14 | 15.03.2022 | Свёрточная операция в подробностях.<br />Конфигурация простейшей свёрточной искусственной нейронной сети. | [видеозапись](https://ml4es.ru/links/ML4ES2-2022-03-15-Lect14.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect14) |
| Занятие 15 | 18.03.2022 | Семинар. Тьюториал. Оптимизация и ускорение процессов подготовки, аугментации данных и передачи на GPU. | [видеозапись](https://ml4es.ru/links/ML4ES2-2022-03-18-Seminar-data-preprocessing.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect15) |
| Занятие 16 | 22.03.2022 | Семинар. Практика. Свёрточная операция своими руками на numpy. | [видеозапись](https://ml4es.ru/links/ML4ES2-2022-03-22-Lect15.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect16) |
| Лекция 17 | 31.03.2022 | Свёрточные нейронные сети: модификации операции свёртки | [видеозапись](https://ml4es.ru/links/ML4ES2-2022-03-31-Lect17.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect17) |
| Лекция 18 | 07.04.2022 | Свёрточные нейронные сети: обзор архитектур;<br />Подходы Transfer Learning, Fine Tuning. | [видеозапись](https://ml4es.ru/links/ML4ES2-2022-04-07-Lect18.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect18) |
| ДЗ №4 | 07.04.2022 | Обзор исследовательской статьи, представляющей архитектуру СНС или подход решения задачи классификации с применением СНС. | [Описание ДЗ](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/HW04) |
| Лекция 19 | 14.04.2022 | Вычисление градиентов в Pytorch.<br />Визуализация и интерпретация свёрточных нейронных сетей. | [видеозапись](https://ml4es.ru/links/ML4ES2-2022-04-14-Lect19.mp4)<br />[материалы занятия](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/tree/master/Lect19) |



### Рекомендуемая литература

- *Гудфеллоу Я., Бенджио И., Курвилль А.* "Глубокое обучение." / М.: ДМК Пресс, 2017. 652 c.
- ***Николенко С. И., Кадурин А. А., Архангельская Е. О.* "Глубокое обучение." / СПб.: Питер. 2019. 480 с.**
- [CS231n Stanford course lecture notes](https://cs231n.github.io/)
- [Deep learning](http://www.deeplearningbook.org/) online book

Дополнительные источники

- *Флах П.* "Машинное обучение. Наука и искусство построения алгоритмов, которые извлекают знания из данных." / Флах П. М.: ДМК Пресс, 2015. 400 c.

- *Шай Шалев-Шварц, Шай Бен-Давид* "Идеи машинного обучения." / - М.: ДМК Пресс, 2018. 432 c.

- [*Hastie T., Tibshirani R., Friedman J.* "The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition"](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) / T. Hastie, R. Tibshirani, J. Friedman, 2-е изд., New York: Springer-Verlag, 2009.

  ​	Первод на русский язык: ([Фридман Дж., Хасти Т., Тибширани Р. "Основы статистического обучения"](http://www.combook.ru/product/11965387/))

- *Bishop C.* "Pattern Recognition and Machine Learning" / C. Bishop, New York: Springer-Verlag, 2006.

  ​	Перевод на русский язык: [Бишоп К.М. "Распознавание образов и машинное обучение"](http://www.combook.ru/product/11965388/)

- **[Matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) - справочник по соотношениям в матричной форме**

- Курс лекций [К.В. Воронцова](http://www.machinelearning.ru/wiki/index.php?title=%D0%A3%D1%87%D0%B0%D1%81%D1%82%D0%BD%D0%B8%D0%BA:%D0%9A%D0%BE%D0%BD%D1%81%D1%82%D0%B0%D0%BD%D1%82%D0%B8%D0%BD_%D0%92%D0%BE%D1%80%D0%BE%D0%BD%D1%86%D0%BE%D0%B2):
  - [Математические методы обучения по прецедентам](http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf)
  - [Оценивание и выбор моделей](http://www.machinelearning.ru/wiki/images/2/2d/Voron-ML-Modeling.pdf)
  - [Логические алгоритмы классификации](http://www.machinelearning.ru/wiki/images/3/3e/Voron-ML-Logic.pdf)
  - [Алгоритмические композиции](http://www.machinelearning.ru/wiki/images/0/0d/Voron-ML-Compositions.pdf)
  
- Курс лекций [Л.М. Местецкого](http://www.machinelearning.ru/wiki/index.php?title=%D0%A3%D1%87%D0%B0%D1%81%D1%82%D0%BD%D0%B8%D0%BA:Mest) ["Математические методы распознавания образов"](http://www.ccas.ru/frc/papers/mestetskii04course.pdf)

- Препринт книги Cosma Rohilla Shalizi "Advanced Data Analysis from an Elementary Point of View". [Доступен онлайн](https://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/).

- *James G., Witten D., Hastie T., Tibshirani R.,* 2013. "An Introduction to Statistical Learning: with Applications in R", Springer Texts in Statistics. Springer-Verlag, New York. Книга [доступна для скачивания](http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf).

- **Интерактивная онлайн-книга *Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola* ["Dive into Deep Learning"](http://d2l.ai/)**

-------

[Подборка](https://towardsdatascience.com/springer-has-released-65-machine-learning-and-data-books-for-free-961f8181f189) книг издательства Springer, доступных для свободной загрузки, по компьютерным наукам, комьютерному зрению, машинному обучению и науке о данных.

[Полный список](https://link.springer.com/search/page/3?facet-content-type="Book"&package=openaccess) книг издательства Springer, выложенных в открытый доступ для свободной загрузки.
