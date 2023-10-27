# DRSO
DRSO (Defensive Response to Scoring Opportunities) (Umemoto *et al.*, StatsBomb Conference 2023): [https://statsbomb.com/wp-content/uploads/2023/10/Evaluation-of-Team-Defense-Positioning-by-Computing-Counterfactuals-Using-StatsBomb-360-Data.pdf](https://statsbomb.com/wp-content/uploads/2023/10/Evaluation-of-Team-Defense-Positioning-by-Computing-Counterfactuals-Using-StatsBomb-360-Data.pdf)

### Propostion
**EF-OBSO**: How to calculate OBSO with Event and Freeze Frame data.<br>
**DRSO**: How to identify a defender' optimal position.<br>

See the following slides if you want to know details.<br>
[2023statsbomb_umemoto_slides.pptx](https://github.com/Rikuhei-ynwa/DRSO/blob/main/2023statsbomb_umemoto_slides.pptx)

## Requirements

* python 3.11
* To install requirements:

```shell
python3 -m pip install -r requirements.txt
```

## Data
* Use StatsBomb opendata: https://github.com/statsbomb/open-data/ (I modified the filename (``open-data-master``) to ``open-data-20231002`` because I identified when I used the cloned data.)
* Place the data under the directory in which this repository resides (see image below).
```
📁 {example}/
├─📁 statsbomb-open-rawdata/
├─📁 DRSO-master/
├─📁 DRSO_data/data-{args.data}/" + args.game
```

## Usage

* See `run.sh`

### Option

- `--count`: Decide how many matches' data are to be processed in parallel when calculating OBSO and DRSO.
- `--data`: Decide which data format to use, default is statsbomb.
- `--game`: Decide which convention data to use, default is all
  + **Caution**: Do not run with the defaults. Also, the option choices are limited to 'wc2022' (World Cup 2022), "euro2020" (UEFA EURO 2020) and 'euro2022' (UEFA EURO 2022). (to use StatsBomb 360 data).
- `--set_vel`: Set the speed of all off-ball players (as the StatsBomb 360 data does not contain information on player speed).
- `--skip_load_rawdata`: Set this if you want to skip loading and preprocessing rawdata.
- `--skip_compare_the_number_of_players`: Set this if you want to skip visualising the reasons I calculated OBSO and DRSO only in the attacking-third.
- `--skip_calculate_obso`: Set this if you want to skip calculating OBSO.
- `--skip_verify_obso`: Set this if you want to skip verifying OBSO.
- `--skip_identify_optimal_positioning`: Set this if you want to skip calculating DRSO.
- `--skip_evaluate_team_defense`: Set this if you want to skip evaluating team defenses.
- `--skip_show_results`: Set this if you want to skip visualising results.
- `--pickle`: Decide which pickle protocol is used, default is 5.

## Reference

```
@article{umemotoevaluation,
  title={Evaluation of Team Defense Positioning by Computing Counterfactuals using StatsBomb 360 data},
  author={Umemoto, Rikuhei and Fujii, Keisuke}
}
```

## License
MIT

