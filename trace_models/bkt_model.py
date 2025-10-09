import numpy as np
from sklearn.metrics import roc_auc_score

class BayesianKnowledgeTracing:
    """
    Bayesian Knowledge Tracing (BKT) implementation using EM for parameter estimation.
    """

    def __init__(self, 
                 p_init=0.1, 
                 p_learn=0.1, 
                 p_slip=0.1, 
                 p_guess=0.1, 
                 min_prob=1e-6):
        self.p_init = p_init
        self.p_learn = p_learn
        self.p_slip = p_slip
        self.p_guess = p_guess
        self.min_prob = min_prob
        self.log_likelihood = None

    # ------------------------------------------------------------------
    #  Forward–Backward Algorithm
    # ------------------------------------------------------------------
    def _forward_backward(self, responses):
        """
        Run the forward–backward algorithm on one sequence of binary responses (0/1).
        Returns posterior of knowledge, pairwise transition posterior, and log-likelihood.
        """
        T = len(responses)
        # Transition matrix (no forgetting)
        transition = np.array([[1 - self.p_learn, self.p_learn],
                               [0.0, 1.0]])
        # Emission matrix: rows=state, cols=observation (0=incorrect, 1=correct)
        emission = np.array([[1 - self.p_guess, self.p_guess],     # if not known
                             [self.p_slip,       1 - self.p_slip]]) # if known

        # Forward pass
        alpha = np.zeros((T, 2))
        initial_knowledge = np.array([1 - self.p_init, self.p_init])
        scale_factors = np.zeros(T)

        alpha[0] = initial_knowledge * emission[:, responses[0]]
        scale_factors[0] = alpha[0].sum() or 1e-12
        alpha[0] /= scale_factors[0]

        for t in range(1, T):
            for next_state in (0, 1):
                alpha[t, next_state] = (alpha[t - 1] @ transition[:, next_state]) * emission[next_state, responses[t]]
            scale_factors[t] = alpha[t].sum() or 1e-12
            alpha[t] /= scale_factors[t]

        # Backward pass
        beta = np.zeros((T, 2))
        beta[-1] = 1.0 / scale_factors[-1]

        for t in range(T - 2, -1, -1):
            next_obs = responses[t + 1]
            for state in (0, 1):
                beta[t, state] = (transition[state] * emission[:, next_obs] * beta[t + 1]).sum()
            beta[t] /= scale_factors[t]

        # Posterior probabilities
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        # Pairwise transition posterior
        xi = np.zeros((max(0, T - 1), 2, 2))
        for t in range(T - 1):
            next_obs = responses[t + 1]
            numerator = (alpha[t][:, None] * transition) * (emission[:, next_obs] * beta[t + 1])[None, :]
            denominator = numerator.sum() or 1e-12
            xi[t] = numerator / denominator

        log_likelihood = np.sum(np.log(scale_factors + 1e-12))
        return gamma, xi, log_likelihood

    # ------------------------------------------------------------------
    #  EM Training
    # ------------------------------------------------------------------
    def fit(self, response_sequences, max_iterations=200, tolerance=1e-4, verbose=False):
        """
        Fit BKT parameters to sequences using the Expectation–Maximization (EM) algorithm.
        """
        prev_log_likelihood = -np.inf

        for iteration in range(max_iterations):
            # Initialize accumulators
            sum_init_known = 0.0
            sum_learn_transitions = 0.0
            sum_unknown_states = 0.0
            sum_known_states = 0.0
            sum_slip_errors = 0.0
            sum_guess_correct = 0.0
            total_loglikelihood = 0.0
            total_responses = 0.0

            for responses in response_sequences:
                gamma, xi, loglik = self._forward_backward(responses)
                total_loglikelihood += loglik
                total_responses += len(responses)

                # E-step accumulations
                sum_init_known += gamma[0, 1]
                sum_known_states += gamma[:, 1].sum()
                sum_slip_errors += (gamma[:, 1] * (responses == 0)).sum()
                sum_unknown_states += gamma[:, 0].sum()
                sum_guess_correct += (gamma[:, 0] * (responses == 1)).sum()
                if len(responses) > 1:
                    sum_learn_transitions += xi[:, 0, 1].sum()

            # M-step: Update parameters
            num_sequences = max(1, len(response_sequences))
            self.p_init = np.clip(sum_init_known / num_sequences, self.min_prob, 1 - self.min_prob)
            self.p_learn = np.clip(sum_learn_transitions / max(sum_unknown_states, 1e-12), self.min_prob, 1 - self.min_prob)
            self.p_slip = np.clip(sum_slip_errors / max(sum_known_states, 1e-12), self.min_prob, 1 - self.min_prob)
            self.p_guess = np.clip(sum_guess_correct / max(sum_unknown_states, 1e-12), self.min_prob, 1 - self.min_prob)

            avg_loglik = total_loglikelihood / total_responses
            if verbose:
                print(f"Iter {iteration:03d}: "
                      f"LL={total_loglikelihood:.4f}, AvgLL={avg_loglik:.4f}, "
                      f"init={self.p_init:.4f}, learn={self.p_learn:.4f}, slip={self.p_slip:.4f}, guess={self.p_guess:.4f}")

            if abs(total_loglikelihood - prev_log_likelihood) < tolerance:
                break
            prev_log_likelihood = total_loglikelihood

        self.log_likelihood = prev_log_likelihood

    # ------------------------------------------------------------------
    #  Filtering and Prediction
    # ------------------------------------------------------------------
    def _forward_filtered(self, responses):
        """
        Forward filtering only — compute P(K_t=1 | responses_1..t).
        """
        T = len(responses)
        transition = np.array([[1 - self.p_learn, self.p_learn],
                               [0.0, 1.0]])
        emission = np.array([[1 - self.p_guess, self.p_guess],
                             [self.p_slip, 1 - self.p_slip]])
        prior = np.array([1 - self.p_init, self.p_init])

        alpha = np.zeros((T, 2))
        scale = np.zeros(T)
        alpha[0] = prior * emission[:, responses[0]]
        scale[0] = alpha[0].sum() or 1e-12
        alpha[0] /= scale[0]

        for t in range(1, T):
            for next_state in (0, 1):
                alpha[t, next_state] = (alpha[t - 1] @ transition[:, next_state]) * emission[next_state, responses[t]]
            scale[t] = alpha[t].sum() or 1e-12
            alpha[t] /= scale[t]

        log_likelihood = np.sum(np.log(scale + 1e-12))
        prob_known = alpha[:, 1].copy()
        return prob_known, log_likelihood

    def predict_one_step(self, prob_known_t):
        """
        Compute one-step-ahead predicted probability of a correct response at t+1.
        """
        T = len(prob_known_t)
        if T < 2:
            return np.array([])
        preds = np.zeros(T - 1)
        for t in range(T - 1):
            p_known_next = prob_known_t[t] + (1 - prob_known_t[t]) * self.p_learn
            preds[t] = (1 - self.p_slip) * p_known_next + self.p_guess * (1 - p_known_next)
        return preds

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, test_sequences):
        """
        Evaluate BKT model on test sequences.
        Returns log-likelihood and AUC.
        """

        all_preds, all_truths = [], []
        total_loglik, total_responses = 0.0, 0

        for seq in test_sequences:
            prob_known, loglik = self._forward_filtered(seq)
            preds = self.predict_one_step(prob_known)
            if len(preds) > 0:
                truths = seq[1:]
                all_preds.append(preds)
                all_truths.append(truths)
                total_loglik += loglik
                total_responses += len(seq)

        all_preds = np.concatenate(all_preds)
        all_truths = np.concatenate(all_truths)

        # Metrics
        avg_loglik = total_loglik / total_responses
        auc = roc_auc_score(all_truths, all_preds)

        return {
            "avg_loglik_per_response": avg_loglik,
            "auc": float(auc)
        }

if __name__ == "__main__":
    # Using 100 largest school's student responses and BKT, 
    # the parameters estimated for Recognize-ER skill are:
    # p_init=0.3136
    # p_learn=0.0273
    # p_slip=0.1195
    # p_guess=0.06364
    
    # Sample test data:
    er_test_data = {}
    with open("/workspaces/kc_tracing_models/sample_data/er_test_data.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split(",")
            stu = line[0]
            stu_response = [int(l) for l in line[1:]]
            er_test_data[stu] = stu_response
    
    me_test_data = {}
    with open("/workspaces/kc_tracing_models/sample_data/me_test_data.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split(",")
            stu = line[0]
            stu_response = [int(l) for l in line[1:]]
            me_test_data[stu] = stu_response

    print("""
    Using 100 largest school's student responses and BKT, 
    the parameters estimated for Recognize-ER skill are:
    p_init=0.3136
    p_learn=0.0273
    p_slip=0.1195
    p_guess=0.06364
    """)
    bkt = BayesianKnowledgeTracing(
        p_init=0.3136,
        p_learn=0.0273,
        p_slip=0.1195,
        p_guess=0.06364
    )
    result = bkt.evaluate(er_test_data.values())
    print("AUC-ROC on test data with ER as optimal strategy: ", round(result['auc'], 4))

    # the parameters estimated for Recognize-ME skill are:
    # p_init=0.2467,
    # p_learn=0.0903,
    # p_slip=0.0945,
    # p_guess=0.0349

    print("""
    the parameters estimated for Recognize-ME skill are:
    p_init=0.2467,
    p_learn=0.0903,
    p_slip=0.0945,
    p_guess=0.0349
    """)
    bkt = BayesianKnowledgeTracing(
        p_init=0.2467,
        p_learn=0.0903,
        p_slip=0.0945,
        p_guess=0.0349
    )
    result = bkt.evaluate(me_test_data.values())
    print("AUC-ROC on test data with ME as optimal strategy:  ", round(result['auc'], 4))