# Thesis 22nd June (ADMM solver)

Problem (5.1) 

* Simple Matrix Trail (Matrices can be seen on top of the code). Following results are given: 

![image-20230621163026780](C:\Users\Shize\AppData\Roaming\Typora\typora-user-images\image-20230621163026780.png)

* Simple 2-dim Random Walk (See Dissertation Draft)

  * Simulate a finite-state 2-dim Markov Random Walk, where $(i, j), i\in \{0, 1, 2\}, j\in\{0,1,2\}$ denotes for each state. 
  * Apply a classical method for function approximation in Reinforcement Learning: Polynomials 
  * Results: L2 Value convergences under $\epsilon = 0.2$

  ![image-20230622150926253](C:\Users\Shize\AppData\Roaming\Typora\typora-user-images\image-20230622150926253.png)

  * Hyper-parameters: 
    * Start at zero variables $(r_0, \omega_0, y_0) = (0, 0, 0)$ with proper dimensions
    * $\epsilon = 0.2$: This is the tolerance for L2-norm $\|\tilde C\omega + \tilde d\|\leq \epsilon$
    * $\mu = 1$: This value must be greater than 0
    * $\tau = 0.0015$: This value should be less than maximal eigenvalue of matrix $\tilde C$, as it was stated on Theorem 5.1
    * Stopping Criteria (Permitted Dual Gap): 0.01
    * Maximal Iteration Numbers: 1500

  
