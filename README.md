# HyperAgent [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fszrlee%2FHyperAgent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

Author: [Yingru Li](https://richardli.xyz), [Jiawei Xu](https://github.com/jiawei415), [Lei Han](https://www.leihan.org), [Zhi-Quan Luo](https://en.wikipedia.org/wiki/Zhi-Quan_Tom_Luo)

This repository contains the official implementation of the **HyperAgent** algorithm, introduced in our ICML 2024 paper [Q-Star Meets Scalable Posterior Sampling: Bridging Theory and Practice via HyperAgent](https://arxiv.org/abs/2402.10228).

For integrating the Generative Pre-trained Transformer (GPT) with HyperAgent, see [szrlee/GPT-HyperAgent](https://github.com/szrlee/GPT-HyperAgent), designed for adaptive foundation models for online decisions.

<img src="figures/2023112801_param_step_4.png" alt="HyperAgent Performance" style="width: 75%;">

- **Data Efficient** ✅: **HyperAgent** achieves human-level performance (1 IQM) with only 15% of the data used by Double-DQN (DDQN, 2016, DeepMind) in 1.5M interactions.
- **Computation Efficient** ✅: **HyperAgent** uses just 5% of the model parameters compared to the 2023 state-of-the-art algorithm ([BBF](https://paperswithcode.com/paper/bigger-better-faster-human-level-atari-with), DeepMind).
- **Ensemble+ Comparison**: Achieves only 0.22 IQM score under 1.5M interactions and requires double the parameters of HyperAgent.

Reference: 
- [1] [Papers with Code - Atari Games 100k](https://paperswithcode.com/sota/atari-games-100k-on-atari-100k)
- [2] [HyperAgent Paper](https://arxiv.org/abs/2402.10228)

## Installation
```bash
cd HyperAgent
pip install -e .
```

## Usage

To reproduce the results for Atari (e.g., Pong):
```bash
sh experiments/start_atari.sh Pong
```

To reproduce the results for DeepSea (e.g., size 20):
```bash
sh experiments/start_deepsea.sh 20
```

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{li2024hyperagent,
  title         = {{Q-Star Meets Scalable Posterior Sampling: Bridging Theory and Practice via HyperAgent}},
  author        = {Li, Yingru and Xu, Jiawei and Han, Lei and Luo, Zhi-Quan},
  booktitle     = {Forty-first International Conference on Machine Learning},
  year          = {2024},
  series        = {Proceedings of Machine Learning Research},
  eprint        = {2402.10228},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2402.10228}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
