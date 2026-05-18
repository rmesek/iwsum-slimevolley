# MARL Implementation of Slime Volleyball

[Slime Volleyball Gym Environment](https://github.com/hardmaru/slimevolleygym)  
Typ tematu: 2 lub 3 w zależności od algorytmu uczenia  
Wybrana dziedzina: Multi-Agent Reinforcement Learning (MARL)  
Skład grupy: Robert Mesek

## Działanie systemu

Celem agenta jest sprawienie, by piłka wylądowała na polu przeciwnika, co powoduje utratę jednego życia przez przeciwnika. Każdy agent zaczyna z pięcioma życiami. Epizod kończy się, gdy któryś z agentów straci wszystkie pięć żyć lub po upływie 3000 kroków czasowych. Agent otrzymuje nagrodę w postaci +1, gdy przeciwnik straci życie, lub -1, gdy to on straci życie.

## Wybór parametrów

(Wstępna propozycja)

### 1. Wybrany algorytm

Wybrany zostanie algorytm [IPPO (Independent Proximal Policy Optimization)](https://docs.agilerl.com/en/latest/api/algorithms/ippo.html). Agenci będą trenowani od zera poprzez grę z samym sobą (_self-play_).

### 2. Atrybuty wejściowe

Do nauki wykorzystamy 12-wymiarowy wektor cech, co pozwoli na znacznie szybszy trening niż w przypadku analizy obrazu:

- Pozycja `x`, `y` oraz prędkość `vx`, `vy` **własnego agenta**.
- Pozycja `x`, `y` oraz prędkość `vx`, `vy` **piłki**.
- Pozycja `x`, `y` oraz prędkość `vx`, `vy` **przeciwnika**.

### 3. Kategorie nagród

Poza standardowymi nagrodami +1 za zdobycie punktu, -1 za utratę życia, możemy spróbować rozważyć dodatkową nagrodę za kontakt z piłką, aby przyspieszyć naukę w początkowych ewolucjach.
