# Locked grouped models on hard negatives

No theta/K was selected or adjusted on these 30 non-violent videos.

| Model | Category | False positives | Rate |
|---|---|---:|---:|
| lstm_original_grouped | sport | 6/15 | 40.00% |
| lstm_original_grouped | danse | 9/10 | 90.00% |
| lstm_original_grouped | calme | 3/5 | 60.00% |
| lstm_original_grouped | overall | 18/30 | 60.00% |
| lstm_augmented_grouped | sport | 5/15 | 33.33% |
| lstm_augmented_grouped | danse | 5/10 | 50.00% |
| lstm_augmented_grouped | calme | 2/5 | 40.00% |
| lstm_augmented_grouped | overall | 12/30 | 40.00% |
| gru_original_grouped | sport | 7/15 | 46.67% |
| gru_original_grouped | danse | 8/10 | 80.00% |
| gru_original_grouped | calme | 2/5 | 40.00% |
| gru_original_grouped | overall | 17/30 | 56.67% |
| gru_augmented_grouped | sport | 5/15 | 33.33% |
| gru_augmented_grouped | danse | 9/10 | 90.00% |
| gru_augmented_grouped | calme | 2/5 | 40.00% |
| gru_augmented_grouped | overall | 16/30 | 53.33% |
