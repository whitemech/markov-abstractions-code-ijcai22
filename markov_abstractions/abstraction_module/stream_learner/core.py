from markov_abstractions.abstraction_module.stream_learner.params import StreamParams

from copy import deepcopy
from math import log, sqrt, pi, inf
from collections import Counter
from pprint import pprint
import graphviz, pickle


class StreamPDFALearner():
    """Stream PDFA Learner."""

    def __init__(self, params: StreamParams):
        self.params = params
        self.graph = Graph(params)
        
        # Counts for each state
        self.t = 0
        self.next_test = dict()
        self.test_count = dict()
        self.state_count = Counter()
        self.state_length_counts = dict()
        self.state_prefix_counts = dict()
        self.state_prefixes = dict()

        # Init counts for initial state
        self.state_length_counts[self.graph.initial_state] = Counter()
        self.state_prefix_counts[self.graph.initial_state] = Counter()
        self.state_prefixes[self.graph.initial_state] = Counter()
        self.next_test[self.graph.initial_state] = self.params.alpha_0
        self.test_count[self.graph.initial_state] = 1

        # Stats
        self.time_merged = list()
        self.time_promoted_safe = list()
        self.time_new_candidate = list()
        self.time_self_tested = list()
        self.equal_test_sketch_sizes = list()
        self.hypotheses = list()

    def consume(self, x):
        # Track if graph changed
        changed = False
        merges = set()
        # Current state to start walking the graph
        current_state = self.graph.initial_state
        # Foreach decomposition of x
        for w, z in self._decompose(x):
            # Get state for word w
            h = w
            if len(h) > 0:
                s_h = self.graph.transitions.get(current_state, {}).get(h[-1])
            else:
                s_h = current_state
            current_state = s_h
            # If s_h is defined
            if s_h is not None and s_h in self.graph.candidate:
                # Insert suffix to the state counts
                self._update_counts(s_h, z)
                # Test
                if self.state_count[s_h] >= self.next_test[s_h]:
                    self.next_test[s_h] = self.params.alpha * self.next_test[s_h]
                    # Test against each safe state
                    for s_safe in self.graph.safe:
                        if s_safe not in self.graph.distinct_from[s_h]:
                                test = self._test(s_h, s_safe, self._delta_i(self.test_count[s_h]))
                                self.test_count[s_h] = self.test_count[s_h] + 1
                                print(f"Test result for s_h={s_h} and s_safe={s_safe} is: {test} (at time {self.t})")
                                if test == 'equal':
                                    print(f"MERGING state cand={s_h} to safe={s_safe} (at time {self.t})")
                                    # Merge s_h to s_safe
                                    self.graph.merge_states(s_h, s_safe)
                                    merges.add((s_h, s_safe))
                                    # Propagate strings
                                    if self.params.string_propagation:
                                        propagated = self._propagate_strings_merge(s_h, s_safe)
                                    # Current state becomes the safe state
                                    current_state = s_safe
                                    # Stats
                                    self.time_merged.append((self.t, (s_h, s_safe)))
                                    changed = True
                                    break
                                elif test == 'distinct':
                                    # Mark s_safe as distinct from s_h
                                    self.graph.distinct_from[s_h].add(s_safe)
                                    self.graph.distinct_from[s_safe].add(s_h)
                    # If s_h has not self-tested equal yet
                    if s_h not in self.graph.self_tested:
                            test = self._test(s_h, s_h, self._delta_i(self.test_count[s_h]))
                            self.test_count[s_h] = self.test_count[s_h] + 1
                            if test == 'equal':
                                print(f"Self-test result for s_h={s_h} is: {test} (at time {self.t})")
                                self.graph.self_tested.add(s_h)
                                # Stats
                                self.time_self_tested.append((self.t, s_h))
                    # If s_h is marked as distinct from all safe states
                    s_h_is_distinct = True
                    for s_safe in self.graph.safe:
                        if s_h not in self.graph.distinct_from[s_safe]:
                            s_h_is_distinct = False
                            break
                    if s_h_is_distinct and s_h in self.graph.self_tested:
                        print(f"Promoting distinct s_h={s_h} to safe with word: {h} (at time {self.t})")
                        # Promote s_h to safe
                        self.graph.promote_to_safe(s_h)
                        # Stats
                        self.time_promoted_safe.append((self.t, s_h))
                        changed = True
                        # Add new candidates w+gamma+sigma for each symbol in the alphabet
                        for gamma in self.params.Gamma:
                            for sigma in self.params.Sigma:
                                symbol = (gamma, sigma)
                                s_new_word = self.graph.add_candidate(s_h, symbol)
                                # print(f"New candidate state for word {new_word} is s={s_new_word}")
                                # Init counts for new state
                                self.state_length_counts[s_new_word] = Counter()
                                self.state_prefix_counts[s_new_word] = Counter()
                                self.state_prefixes[s_new_word] = Counter()
                                # Start test information
                                self.test_count[s_new_word] = 1
                                self.next_test[s_new_word] = self.params.alpha_0
                                # Propagate strings
                                if self.params.string_propagation:
                                    propagated = self._propagate_strings(s_h, s_new_word, symbol)
                                # Stats
                                self.time_new_candidate.append((self.t, s_new_word))

        # Increment time
        self.t += 1

        # If changed, save hypothesis
        if changed: 
            # Save hypothesis
            self.graph.to_dot(
                filename=f"hyp_{self.t}", directory=f"{self.params.log_path}/abstraction/hypotheses"
            )

        # Return merges
        return merges, self.graph

    def _propagate_strings_merge(self, state_from, state_to):
        total_propagated = 0
        for string, count in self.state_prefixes[state_from].items():
            curr_state = state_to
            for pos, symbol in enumerate(string):
                curr_state = self.graph.transitions.get(curr_state, {}).get(symbol)
                if curr_state is not None and curr_state in self.graph.candidate:
                    # Cut first symbol of the string (the one that leads to the candidate)
                    z = string[pos+1:]
                    # Propagate state count and prefixes
                    self.state_count.update([curr_state]*count)
                    self.state_prefixes[curr_state].update([z]*count)
                    # Cut z if necessary
                    if len(z) > self.params.d:
                        z = z[:self.params.d]
                    for i in range(len(z)+1):
                        z_i = z[:i]
                        self.state_length_counts[curr_state].update([i]*count)
                        self.state_prefix_counts[curr_state].update([z_i]*count)
                    total_propagated += count
                    # Break (go for next string)
                    break
        return total_propagated

    def _propagate_strings(self, safe, candidate, symbol):
        total_propagated = 0
        for string, count in self.state_prefixes[safe].items():
            if len(string) > 0 and string[0] == symbol:
                # Cut first symbol of the string (the one that leads to the candidate)
                z = string[1:]
                # Propagate state count and prefixes
                self.state_count.update([candidate]*count)
                self.state_prefixes[candidate].update([z]*count)
                # Cut z if necessary
                if len(z) > self.params.d:
                    z = z[:self.params.d]
                for i in range(len(z)+1):
                    z_i = z[:i]
                    self.state_length_counts[candidate].update([i]*count)
                    self.state_prefix_counts[candidate].update([z_i]*count)
                total_propagated += count
        return total_propagated

    def _test(self, s_cand, s_safe, delta_i):
        """Test candidate s_cand and s_safe."""
        # Useful quantitites
        d = self.params.d
        nb_A = self.params.nb_actions
        nb_O = self.params.nb_observations
        nb_R = self.params.nb_rewards

        # Compute distances for each length i in [0,d]
        mu_U = set()
        mu_L = set()
        ##
        max_mu_U = -1
        max_mu_L = -1
        ##
        for i in range(d+1):
            # Compute M quantities
            cand_count = self.state_length_counts[s_cand][i]
            safe_count = self.state_length_counts[s_safe][i]
            if cand_count == 0 or safe_count == 0:
                mu_hat, max_prefix, max_cand_freq, max_safe_freq = inf, inf, inf, inf
            else:
                M_quantity = cand_count + safe_count
                M_p_quantity = (cand_count * safe_count) / ((sqrt(cand_count) + sqrt(safe_count))**2)
                # Compute Delta
                Delta = sqrt((1 / (2 * M_p_quantity)) * log((4 * d * (nb_A*nb_O*nb_R)**i) / (delta_i)))
                # Compute prefix L infinity distance
                mu_hat, max_prefix, max_cand_freq, max_safe_freq = self._compute_prefix_L_infinity(s_cand, s_safe, i)
                # mu_hat = self._compute_prefix_L_infinity(s_cand, s_safe, i)

                mu_hat_upper = mu_hat + Delta
                mu_hat_lower = mu_hat - Delta

                # print(self.t, 'len', i, 'Mp', M_p_quantity, '\tdelta i', delta_i, '\tDelta', Delta, '\tMu_hat', mu_hat)
                # print(max_prefix, '\n', max_cand_freq, max_safe_freq, '\n')

                mu_U.add(mu_hat_upper)
                mu_L.add(mu_hat_lower)

            ##
            if mu_hat_upper > max_mu_U:
                max_mu_U = mu_hat_upper
                Delta_for_mu_U = Delta
                mu_hat_for_mu_U = mu_hat
                length_i_for_mu_U = i
                prefix_for_mu_U = max_prefix
                prefix_freq_cand_for_mu_U = max_cand_freq
                prefix_freq_safe_for_mu_U = max_safe_freq
            if mu_hat_lower > max_mu_L:
                max_mu_L = mu_hat_lower
                Delta_for_mu_L = Delta
                mu_hat_for_mu_L = mu_hat
                length_i_for_mu_L = i
                prefix_for_mu_L = max_prefix
                prefix_freq_cand_for_mu_L = max_cand_freq
                prefix_freq_safe_for_mu_L = max_safe_freq
            ##

        # Get max for mu_U and mu_L
        mu_hat_upper = max(mu_U)
        mu_hat_lower = max(mu_L)

        if mu_hat_upper < self.params.mu:
            # Stats
            self.equal_test_sketch_sizes.append((self.t, (s_cand, cand_count), (s_safe, safe_count)))

            # print()
            # print()
            # print("EQUAL") 
            # print('--')
            # print(f'mu: {self.params.mu}') 
            # print(f'mu_upper: {mu_hat_upper}') 
            # print(f'mu_lower: {mu_hat_lower}') 
            # print(f'delta_i: {delta_i}')
            # print('--')
            # print(f'Delta_for_mu_U: {Delta_for_mu_U}')
            # print(f'mu_hat_for_mu_U: {mu_hat_for_mu_U}')
            # print(f'length_i_for_mu_U: {length_i_for_mu_U}')
            # print(f'prefix_for_mu_U: {prefix_for_mu_U}')
            # print(f'cand={s_cand} prefix_freq_for_mu_U: {prefix_freq_cand_for_mu_U}')
            # print(f'safe={s_safe} prefix_freq_for_mu_U: {prefix_freq_safe_for_mu_U}')
            # print('--')
            # print(f'Delta_for_mu_L: {Delta_for_mu_L}')
            # print(f'mu_hat_for_mu_L: {mu_hat_for_mu_L}')
            # print(f'length_i_for_mu_L: {length_i_for_mu_L}')
            # print(f'prefix_for_mu_L: {prefix_for_mu_L}')
            # print(f'cand={s_cand} prefix_freq_for_mu_L: {prefix_freq_cand_for_mu_L}')
            # print(f'safe={s_safe} prefix_freq_for_mu_L: {prefix_freq_safe_for_mu_L}')
            # print('--')
            # print(f'Count for cand={s_cand}: {cand_count}')
            # print(f'Count for safe={s_safe}: {safe_count}')

            # Then decide that S = S'
            return 'equal'
        elif mu_hat_lower > 0:

            # print()
            # print()
            # print("DISTINCT") 
            # print('--')
            # print(f'mu: {self.params.mu}') 
            # print(f'mu_upper: {mu_hat_upper}') 
            # print(f'mu_lower: {mu_hat_lower}') 
            # print(f'delta_i: {delta_i}')
            # print('--')
            # print(f'Delta_for_mu_U: {Delta_for_mu_U}')
            # print(f'mu_hat_for_mu_U: {mu_hat_for_mu_U}')
            # print(f'length_i_for_mu_U: {length_i_for_mu_U}')
            # print(f'prefix_for_mu_U: {prefix_for_mu_U}')
            # print(f'cand={s_cand} prefix_freq_for_mu_U: {prefix_freq_cand_for_mu_U}')
            # print(f'safe={s_safe} prefix_freq_for_mu_U: {prefix_freq_safe_for_mu_U}')
            # print('--')
            # print(f'Delta_for_mu_L: {Delta_for_mu_L}')
            # print(f'mu_hat_for_mu_L: {mu_hat_for_mu_L}')
            # print(f'length_i_for_mu_L: {length_i_for_mu_L}')
            # print(f'prefix_for_mu_L: {prefix_for_mu_L}')
            # print(f'cand={s_cand} prefix_freq_for_mu_L: {prefix_freq_cand_for_mu_L}')
            # print(f'safe={s_safe} prefix_freq_for_mu_L: {prefix_freq_safe_for_mu_L}')
            # print('--')
            # print(f'Count for cand={s_cand}: {cand_count}')
            # print(f'Count for safe={s_safe}: {safe_count}')

            # Then decide that S != S'
            return 'distinct'
        else:
            # Then answer is 'I don't know'
            return 'unknown'

    def _compute_prefix_L_infinity(self, s_cand, s_safe, prefix_length): 
        # Get prefixes of current length i
        all_prefixes = set(self.state_prefix_counts[s_cand].keys()).union(set(self.state_prefix_counts[s_safe].keys()))
        prefixes = set()
        for prefix in all_prefixes:
            if len(prefix) == prefix_length:
                prefixes.add(prefix)
        
        ##
        max_dist = 0
        max_prefix = ''
        max_cand_freq = 0
        max_safe_freq = 0
        ##
        # Compute distances
        distances = list([0])
        for prefix in prefixes:
            cand_freq = self.state_prefix_counts[s_cand][prefix] / self.state_count[s_cand]
            safe_freq = self.state_prefix_counts[s_safe][prefix] / self.state_count[s_safe]
            dist = abs(cand_freq - safe_freq)
            distances.append(dist)
            ##
            if dist > max_dist:
                max_dist = dist
                max_prefix = prefix
                max_cand_freq = cand_freq
                max_safe_freq = safe_freq
            ##
        mu_hat = max(distances)
        # return mu_hat
        return mu_hat, max_prefix, max_cand_freq, max_safe_freq

    def _update_counts(self, s, z):
        self.state_count.update([s])
        self.state_prefixes[s].update([z])
        if len(z) > self.params.d:
            z = z[:self.params.d]
        for i in range(len(z)+1):
            z_i = z[:i]
            self.state_length_counts[s].update([i])
            self.state_prefix_counts[s].update([z_i])

    def _delta_i(self, i):
        delta_i = (6 * self.params.delta_p) / (pi**2 * i**2)
        return delta_i

    def _decompose(self, x):
        """Decompose a word into x = wz."""
        decompositions = list()
        for i in range(len(x)+1):
            w = tuple(x[:i])
            z = tuple(x[i:])
            decompositions.append((w, z))
        return decompositions
    
    def _project_word_on_observations(self, word):
        return tuple([trans[1] for trans in word])
    
    def get_stats(self):
        stats = dict()
        stats['time_merged'] = self.time_merged
        stats['time_promoted_safe'] = self.time_promoted_safe
        stats['time_new_candidate'] = self.time_new_candidate
        stats['time_self_tested'] = self.time_self_tested
        stats['equal_test_sketch_sizes'] = self.equal_test_sketch_sizes
        return stats


class Graph:
    """Represent a PDFA subgraph."""

    def __init__(self, params: StreamParams):
        """Initialize."""
        self.params = params

        # Vertices
        self.initial_state = 0
        self.last_vertex = self.initial_state
        self.vertices = set([self.initial_state])
        
        # Transitions {state: {char: next_state}}
        self.transitions = {self.initial_state: {}}

        # Set of safe, candidate, and insignificant states
        self.safe = set()
        self.candidate = set([self.initial_state])
        self.insignificant = set()
        self.distinct_from = dict({self.initial_state: set()})
        self.self_tested = set()

    def promote_to_safe(self, q):
        self.safe.add(q)
        self.candidate.remove(q)

    def merge_states(self, q_from, q_to):
        """Merge candidate q_from to safe q_to."""
        # Remove candidate from sets
        self.vertices.remove(q_from)
        self.candidate.remove(q_from)
        self.distinct_from[q_to] = self.distinct_from[q_to].union(self.distinct_from[q_from])
        # Transitions to q_from now go to q_to
        for state_from, trans in self.transitions.items():
            for character, state_to in trans.items():
                if state_to == q_from:
                    self.transitions[state_from][character] = q_to
        del self.transitions[q_from]

    def add_vertex(self, prev_state, symbol):
        new_vertex = self.last_vertex + 1
        self.last_vertex += 1
        self.vertices.add(new_vertex)
        self.distinct_from[new_vertex] = set()
        self.add_transition(new_vertex, prev_state, symbol)
        return new_vertex

    def add_candidate(self, prev_state, symbol):
        new_vertex = self.add_vertex(prev_state, symbol)
        self.candidate.add(new_vertex)
        return new_vertex
        
    def add_transition(self, new_vertex, prev_state, symbol):
        # NOTE: Candidate and insignificant states have no outgoing transitions
        if prev_state in self.safe:
            self.transitions[new_vertex] = {}
            self.transitions[prev_state][symbol] = new_vertex

    def to_dot(
        self, filename: str = "hyp", directory: str = ".", show_candidates: bool = False
    ) -> str:
        print(f"Saving DOT to {directory}/{filename}")
        graph = graphviz.Digraph(format="dot")
        graph.node("fake", style="invisible")
        for state in self.vertices:
            if state == 0:
                graph.node(str(state), root="true")
            else:
                if state in self.candidate and show_candidates:
                    graph.node(str(state), shape="square", style="dashed", color="gray")
                elif state not in self.candidate:
                    graph.node(str(state))
        graph.edge("fake", str(0), style="bold")
        for start, char2end in self.transitions.items():
            for char, end in char2end.items():
                if start not in self.candidate and end in self.candidate and show_candidates:
                    graph.edge(
                        str(start),
                        str(end),
                        label=f"{str(char)}",
                        style="dashed",
                        color="gray"
                    )
                elif start not in self.candidate and end not in self.candidate:
                    graph.edge(
                        str(start),
                        str(end),
                        label=f"{str(char)}"
                    )
        
        # Render and save
        graph.save(filename=filename + ".dot", directory=directory)
        print("Finished")
