## Revision

### Reviewer A

1.  I don't understand why the beta does not appear in Eq. (9) because the authors' p(s|J) has beta. I think that Eqs. (9-11) should be rechecked for mathematical rigor, although the beta should be irrelevant in the end for maximization of the log-likelihood function.

- Re : The major purpose of inverse Ising problems is to infer for network structure such as protein network, stock network and these networks are difficult to define temperature. Thus the beta is conventionally excluded from the Boltzmann distribution, in inverse Ising problem.

2. The authors said in the Introduction that "Without much consideration, one can expect that EM performance is maximized near the critical temperature for the phase transition." This statement needs to be elaborated more in the Introduction to deliver a clear idea to readers about what motivates this work.

- Re : 

### Reviwer B

1. As they mentioned, the reweighting parameter epsilon of the erasure machine can be interpreted as a temperature-like parameter. I worry that ¡°temperature-dependent performance¡± in the title and texts may confuse readers as if this study examined the effect of the hyperparameter epsilon for the performance. Actually, they studied how well the erasure machine inferred interaction strengths from Monte-Carlo samples generated at different temperatures. This should be clarified in the title and texts. 

- Re : OK

2. The performance of the erasure machine depends on the hyperparameter epsilon, and learning rate alpha. It is questionable how sensitive their findings are for these parameters. In addition, the number of iteration in the erasure machine is crucial for accurate inference because too much iteration sometimes leads to overestimation. Therefore, the optimality of the iteration should be checked more carefully. In particular, this may affect their conclusion on the speculation regarding the critical temperature (please refer the following issue). 

- Re : 우리도 연구 초기에 하이퍼파라리터에 대한 영향을 체크했다. 논문에 언급한대로 하이퍼파라미터의 영향은 적었으며, 모든 시뮬레이션에서 동일하게 적용되었다.\\
그럼에도 불구하고 영향이 아예없지는 않은데, 그림을 첨부한다.

3. It is expected that the erasure machine works best for samples generated at an intermediate temperature somewhere between low and high temperature. However, it is not clear why they speculate that the optimal temperature may correspond to the critical temperature. I wonder the motivation for this speculation. 

- Re : 샘플 complexity를 생각해보면, 간단한 형태의 PDF를 가지고 있는 high temperature sample 과 low temperature sample 은 학습이 잘 되어지지 않을것 이다.\\
그렇다면, 어느 중간 온도에서 학습이 가장 잘 될 텐데, criteristic temperature 가 Tc 말고는 없기 때문에 Tc에서 학습이 잘 될 것이라고 생각했다.

4. They observed that as a system gets larger, better inference resulted from samples generated at a higher temperature than the critical temperature. They mentioned that this was unexpected since they expect that their speculation would fit better at the thermodynamic limit with a larger system size. My opinion is different. For equivalent quality of inference for a larger system, more samples are usually required than a smaller system. Thus the deviation from their expectation may originate from the relative insufficiency of samples for the larger system. I suggest the authors explore their speculation more carefully with a larger system size. In Fig. 4(a), as sample size is larger, optimal inference seems to occur near the critical temperature. Furthermore, their speculation also seems to be right for the Ising model case in Fig. 7. I wonder if their best inference approaches closer into the critical temperature when they use a larger system size than M = 5000. 

- Re : 우리는 너의 의견대로 시스템 사이즈가 커지면, 샘플사이즈가 모자라져서 ~어쩌구~ 라는것에 동의한다. \\
우리도 그러한 생각을 했지만 시스템 사이즈를 키우는 것에는 시간적 복잡도가 너무 커서 L=320 이상 키우는 것에는 무리가 있습니다. \\
따라서 우리는 당신과 같이 주장하는것에 주저됩니다.

5. The authors need to comment on the effectiveness of the regularization. Why does the Ising model show better performance with the L1 regularization and the spin glass model show better performance with the L2 regularization?

- Re : OK

6. The following sentence in Summary and Discussion is difficult to understand. "The reason can be that the EM framework basically set the prior distribution to a uniform distribution." Detailed explanation is necessary.

-Re : OK
