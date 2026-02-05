# SKOR AKURASI VALIDASI



### ACCURACY:



##### Facenet Pretrained:



NB: PRETRAINED



--- Starting 5-Fold Subject-Independent Cross Validation ---

Found 8 Subjects: \[5, 8, 9, 11, 12, 15, 17, 18]

Device: cuda



=== Fold 1/5 ===

Train Subjects: 6, Test Subjects: 2

Train Files: 842, Test Files: 119

Fold 1 Best Accuracy: 91.60%



=== Fold 2/5 ===

Train Subjects: 6, Test Subjects: 2

Train Files: 484, Test Files: 477

Fold 2 Best Accuracy: 68.97%



=== Fold 3/5 ===

Train Subjects: 6, Test Subjects: 2

Train Files: 854, Test Files: 107

Fold 3 Best Accuracy: 99.07%



=== Fold 4/5 ===

Train Subjects: 7, Test Subjects: 1

Train Files: 953, Test Files: 8

Fold 4 Best Accuracy: 100.00%



=== Fold 5/5 ===

Train Subjects: 7, Test Subjects: 1

Train Files: 711, Test Files: 250

Fold 5 Best Accuracy: 100.00%





=== Cross Validation Results ===

Fold 1: 91.60%

Fold 2: 68.97%

Fold 3: 99.07%

Fold 4: 100.00%

Fold 5: 100.00%

---------------------------

Mean Accuracy: 91.93% (+/- 11.90%)



##### CELEBA:



--- Starting 5-Fold Subject-Independent Cross Validation (CelebA) ---

Found 8 Subjects: \[5, 8, 9, 11, 12, 15, 17, 18]

Device: cuda



=== Fold 1/5 ===

Train Subjects: 6, Test Subjects: 2

Train Files: 842, Test Files: 119

Fold 1 Best Accuracy: 87.39%



=== Fold 2/5 ===

Train Subjects: 6, Test Subjects: 2

Train Files: 484, Test Files: 477

Fold 2 Best Accuracy: 58.70%



=== Fold 3/5 ===

Train Subjects: 6, Test Subjects: 2

Train Files: 854, Test Files: 107

Fold 3 Best Accuracy: 94.39%



=== Fold 4/5 ===

Train Subjects: 7, Test Subjects: 1

Train Files: 953, Test Files: 8

Fold 4 Best Accuracy: 62.50%



=== Fold 5/5 ===

Train Subjects: 7, Test Subjects: 1

Train Files: 711, Test Files: 250

Fold 5 Best Accuracy: 98.40%



### BENCHMARK VIDEO:



FACENET:



========================================

 SESSION SUMMARY (8.3s)

========================================

Total Frames Analyzed: 114

Neutral             : 49.1%

Disgust             : 7.0%

Surprise            : 7.0%

Happiness           : 21.1%

Fear                : 0.9%

Anger               : 14.9%

========================================





CELEBA:



========================================

&nbsp;SESSION SUMMARY (5.9s)

========================================

Total Frames Analyzed: 114

Anger               : 14.9%

Disgust             : 10.5%

Fear                : 0.0%

Happiness           : 16.7%

Neutral             : 51.8%

Others              : 0.0%

Repression          : 0.0%

Sadness             : 0.0%

Surprise            : 6.1%

Unknown             : 0.0%

========================================



