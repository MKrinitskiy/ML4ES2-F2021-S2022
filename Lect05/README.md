## ML4ES2, Лекция 5

#### Градиентная оптимизация нейросетей: расписание шага обучения



Лекция посвящена особенностям градиентной оптимизации функции потерь искусственных нейронных сетей.

В частности, затронут вопрос влияния расписания шага обучения (learning rate) на скорость и стабильность обучения.

Обсуждаются возможные варианты расписания шага обучения и их эффекты при тренировке глубоких нейросетей.



Записи к лекции доступны в [PDF](https://github.com/MKrinitskiy/ML4ES2-F2021-S2022/blob/main/Lect05/ML4ES2-2021-11-02-Lect05-notes.pdf)



В качестве материалов на дополнительное чтение можно обратить внимание на следующие источники:

Разбор некоторых расписаний шага обучения: [link](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1)

[Описание](https://fastai.github.io/timmdocs/SGDR) одного из самых интересных и потенциально наиболее продуктивных расписаний - SGDR (Stochastic Gradient Descent with warm Restarts), представленного в [статье на arxiv.org](https://arxiv.org/abs/1608.03983).

[Программная реализация](https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers) некоторых расписаний шага обучения на Pytorch.

