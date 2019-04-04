import numpy as np
from collections import defaultdict

class NBLearner:
    """Naive Bayes learner with Laplace smoothing.
    """
    
    def __init__(self):
        # Model stores prior and posterior probabilities of classes, taking the form -
        # @model: { class_type: ( class_prior_prob: float, class_posterior_probs: [ { attrib_val: (prob, freq) } ] ) } }
        self.model = None
        self.num_instances = None
    
    def train(self, instances, instance_classes, missing_val = ""):
        """Calculate prior and posterior (Laplace smoothed) probabilities from a list of instances 
           and corresponding classes, storing calculated probabilities in 'self.model'.
        """
        
        ### helper functions ###
        
        def count_attrib_vals(instances, instance_classes, class_vals, missing_val):
            """
            @param instances: list of attribute values - [[ attrib_val ]]
            @param instance_classes: list of class values, each element corresponding
                                      to an instance in @param instances
            @param class_vals: list of class values - [ class_val ]
            @return: { class_val: [attribute_index: {attribute_val: frequency}]}
            """
            # Initialise attribute value frequency counter.
            attribute_value_counts = {}
            num_attribs = len(instances[0])

            for class_val in class_vals:
                attribute_value_counts[class_val] = [defaultdict(int) for attrib_i in range(num_attribs)]

            ### Fill attribute value frequency counter.
            for instance_i in range(len(instances)):
                for attrib_i in range(num_attribs):
                    instance_class = instance_classes[instance_i]
                    instance_attrib_val = instances[instance_i][attrib_i]
                    
                    # missing values do not contribute to attribute-value probability estimate
                    if instance_attrib_val != missing_val:
                        # add one to attribute value count of the class of the current instances
                        attribute_value_counts[instance_class][attrib_i][instance_attrib_val] += 1

                        # add 0 to attribute value count of all other classes - this is done so that 
                        # classes are aware of attribute values, even if they only have a count of 0 
                        # (see 'calculate_posterior_laplace_probs')
                        for class_val in class_vals:
                            attribute_value_counts[class_val][attrib_i][instance_attrib_val] += 0

            return attribute_value_counts
        
        def laplace_prob(class_freq, class_attrib_val_freq, num_attrib_vals):
            return (1 + class_attrib_val_freq) / (num_attrib_vals + class_freq)
        
        def calculate_posterior_laplace_probs(attribute_value_counts, instances, instance_classes):
            
            # Calculate laplace approximated posterior frequencies
            laplace_probs = {}
            num_attribs = len(instances[0])
            
            # initialise structure storing laplace probs. Takes form -
            # laplace_probs = { class_val: [ { attrib_val: (prob, freq) } ]}
            for class_val in instance_classes:
                laplace_probs[class_val] = [{} for attrib_i in range(num_attribs)]

            # for each class
            for class_val in attribute_value_counts:
                # for attribute i
                for attrib_i in range(len(attribute_value_counts[class_val])):
                    # for each (value, value frequency) of attribute i
                    for attrib_value, attrib_val_count in attribute_value_counts[class_val][attrib_i].items():
                        # fill laplace value
                        laplace_probs[class_val][attrib_i][attrib_value] = \
                            (laplace_prob(class_count_dict[class_val], 
                                         attrib_val_count, 
                                         len(attribute_value_counts[class_val][attrib_i]))
                            , attrib_val_count)
                            
            return laplace_probs
        
        ### function body ###
        
        num_instances = len(instances)
        assert(num_instances == len(instance_classes) and num_instances != 0)
        
        num_attribs = len(instances[0])
        assert(num_attribs != 0)
        
        # count class frequencies
        class_count_dict = defaultdict(int) 
        for instance_class in instance_classes:
            class_count_dict[instance_class] += 1
        class_counts = class_count_dict.items()
        
        prior_class_probs = { class_val: val_count / len(instance_classes) for class_val, val_count in class_counts }
        
        attribute_value_counts = count_attrib_vals(instances, instance_classes, class_count_dict.keys(), missing_val)
        posterior_laplace_probs = calculate_posterior_laplace_probs(attribute_value_counts, instances, instance_classes)
        
        # store probabilities in NB learner as trained model
        self.model = dict(list(pair_keys_vals(prior_class_probs, posterior_laplace_probs)))
        self.num_instances = num_instances
        return self
        
        
        
    def predict(self, test_instance, missing_val = ""):
        """Predict the class of @param 'test_instance', by calculating the class which
           'test_instance' has the highest probability of belonging to (using Bayes rule
           and assumption of conditional independence).
        """
        
        # (class_val, probability)
        most_likely_class = (None, 0)
        
        for class_type, (class_prob, class_posteriors) in self.model.items():
            posterior_product = 1
            for attrib_i in range(len(test_instance)):
                # a missing value is not considered in probability
                if test_instance[attrib_i] != missing_val:
                    try:
                        posterior_product *= class_posteriors[attrib_i][test_instance[attrib_i]][0]
                    except KeyError:
                        # if an attribute value is in the test instance but has not been seen by the
                        # model, then it does not contribute to the product of the posterior probabilities
                        pass
                        
            
            # don't divide as we normally would when applying Bayes rule, since all probablities
            # will be divided by the same value (P(instance))
            unnormalised_class_prob = class_prob * posterior_product
            if (unnormalised_class_prob > most_likely_class[1]):
                most_likely_class = (class_type, unnormalised_class_prob)
                
        return most_likely_class[0]

def entropy(prob_array):
    return -sum(map(lambda pr: 0 if pr == 0 else pr * np.log2(pr), prob_array))

def info_gain(trained_nb_learner, attrib_i):
    """Calculate information gain (with respect to class) of attribute at index @param attrib_i,
    """
    assert(trained_nb_learner.model != None)
    
    total_inst_num = trained_nb_learner.num_instances

    # initialise data structure for calculating mean info
    
    # @attr_class_freqs: { attrib_val: (attrib_freq, { class_val: class_val_freq }) }
    attr_class_freqs = defaultdict(lambda: (0, {}))
    
    for cls_val, prob_tuple in trained_nb_learner.model.items():
        for attrib_val, val_prob_and_freq in prob_tuple[1][attrib_i].items():
            attr_freq_for_cls = val_prob_and_freq[1]
            
            # increment attribute value frequency
            attr_class_freqs[attrib_val] = (attr_class_freqs[attrib_val][0] + attr_freq_for_cls, 
                                               attr_class_freqs[attrib_val][1]
                                            )
            # increment class frequency given this attribute val
            attr_class_freqs[attrib_val][1][cls_val] = attr_freq_for_cls
    
    # calculate mean info of classes, given knowledge of attribute_i
    mean_info = 0
    for attr_val, freq_and_cls_dict in attr_class_freqs.items():
        
        attr_val_num_instances = sum(freq_and_cls_dict[1].values())
        attr_val_cls_probs = \
            list(map(lambda attr_val_cls_freq: attr_val_cls_freq / attr_val_num_instances, freq_and_cls_dict[1].values()))
        attr_val_cls_entropy = entropy(attr_val_cls_probs)
    
        mean_info += (freq_and_cls_dict[0] / total_inst_num) * (attr_val_cls_entropy)
        
    # calculate class prior entropy
    class_prior_probs = [class_prob_tuple[0] for class_val, class_prob_tuple in trained_nb_learner.model.items()]
    class_entropy = entropy(class_prior_probs)

    return class_entropy - mean_info