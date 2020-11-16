# Word-Sense-Disambiguation
Les ambiguïtés lexicales font parties des langues naturelles, pour exemple, le mot avocat qui peut être à la fois un fruit, mais aussi une personne de loi. Heureusement, les êtres humains ont la capacité, sans trop d'efforts, de désambiguïser un mot en s'aidant du contexte dans lequel il apparait.

Cependant, en Traitement Automatique des Langues Naturelles (TALN), il est plus difficile pour un système automatique de désambiguïser un mot polysémique. Il est donc indispensable d'être capable de différencier deux mots, a priori identique, mais qui dans des contextes différents, ont des significations tout à fait différentes.

Pour résoudre ce problème, il existe deux types de méthodes :  l'induction de sens (Word Sense Induction, WSI) et l'acquisition automatique de sens (Word Sense Desambiguation, WSD).

L'induction de sens consiste à déterminer le sens d'un mot lorsqu'on a pas de connaissance et que le nombre de sens associé à ce mot est inconnu. On essaye donc d'associer les occurrences d'un mot à un sens et d'identifier le nombre de sens d'un mot que le système découvre.

L'acquisition de sens est une méthode à base de connaissances, elle s'appuie donc sur des ressources lexicales. Il existe plusieurs types de ressources : les inventaires de sens qui pour chaque mot associent une liste de sens possible, par exemple un dictionnaire, et des ressources lexicales telle que WordNet qui permet de représenter les différents liens que peuvent avoir plusieurs mots entre eux.

Ce projet étudiant se réalise dans le cadre du projet PolysEmY,  qui a pour but de désambiguïser un ensemble connu d'acronymes dont les différents sens sont connus. 

Sont utilisées les jeux de données francaises fournies par SemEval lors de sa campagne de 2013, qui proposait une tâche WSD : la tâche 12 (https://www.cs.york.ac.uk/semeval-2013/task12.html).
