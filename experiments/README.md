# Experiments

Ce dossier contient vos expérimentations et notebooks de travail en cours.

## Structure

```
experiments/
├── TEST_PER/           # Vos tests et entraînements actuels
│   ├── notebooks/      # Notebooks d'expérimentation
│   ├── configs/        # Configurations d'entraînement
│   └── results/        # Résultats temporaires
└── autres_tests/       # Autres expérimentations
```

## Utilisation

Ce dossier est votre espace de travail pour :
- Tester de nouveaux hyperparamètres
- Expérimenter différentes architectures
- Valider des hypothèses
- Prototypage rapide

Une fois qu'une expérimentation est validée :
1. Déplacer les scripts finalisés vers `src/`
2. Documenter les résultats dans `benchmarks/`
3. Archiver ou nettoyer `experiments/`

## Note

Le contenu de ce dossier peut être désordonné - c'est normal !
C'est un espace de travail, pas de production.
