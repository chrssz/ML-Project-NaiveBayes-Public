# Project: Car Evaluation using Naive Bayes
# Date: 11-10-2022
# Author: Christopher Castillo
# Email: 007819702@coyote.csusb.edu
import os
from collections import defaultdict
#splits dataset by class values. returns dictionary
def split_by_class(data):
    dict = defaultdict(list)
    for x in data:
        if "unacc" in x:
            x.remove("unacc")
            dict["unacc"].append(x)
        elif "acc" in x:
            x.remove("acc")
            dict["acc"].append(x)
        elif "good" in x:
            x.remove("good")
            dict["good"].append(x)
        elif "vgood" in x:
            x.remove("vgood")
            dict["vgood"].append(x)
        elif "tbd" in x:
            x.remove("tbd")
            dict["tbd"].append(x)
    return dict
#prior probability of attribute
def prior_attribute(data,attribute_c,attribute):
    count = 0
    if attribute_c == "buying": pos = 0
    elif attribute_c == "maint": pos=1
    elif attribute_c == "doors": pos=2
    elif attribute_c == "persons": pos=3
    elif attribute_c == "lug_boot": pos=4
    elif attribute_c == "safety": pos=5
    for x in data["unacc"]:
        if attribute in x[pos]:
            count+=1
    for x in data["acc"]:
        if attribute in x[pos]:
            count+=1
    for x in data["good"]:
        if attribute in x[pos]:
            count+=1
    for x in data["vgood"]:
        if attribute in x[pos]:
            count+=1
    return count / (len(data["unacc"]) + len(data["acc"])+ len(data["good"])+len(data["vgood"]))
#prior probability of class
def prior_class(data,m_class):
    return len(data[m_class]) / (len(data["unacc"]) + len(data["acc"])+ len(data["good"])+len(data["vgood"]))

# likelihood of given params
def likelihood(data,attribute_c,attribute,m_class):
    count = 0
    if attribute_c == "buying": pos = 0
    elif attribute_c == "maint": pos=1
    elif attribute_c == "doors": pos=2
    elif attribute_c == "persons": pos=3
    elif attribute_c == "lug_boot": pos=4
    elif attribute_c == "safety": pos=5
    #elif attribute_c == "tbd": pos=6
    for x in data[m_class]:
        if attribute in x[pos]:
            count +=1
    return count / len(data[m_class])                   
def acc(data,training):
    #true positive
    tp = sum(x[6] == y[6] for x, y in zip(data, training))
    #true negative
    tn = sum(x[6] != y[6] for x, y in zip(data, training))
    acc = tp / (tp + tn) * 100
    return acc
def main():
    trainingdata = []
    testdata = []
    with open("cartest.data") as file:
        for line in file: 
            line = line.split(",")
            line[-1] = line[-1].strip() #removes /n from list
            testdata.append(line) 
    with open("car.data") as file:
        for line in file: 
            line = line.split(",")
            line[-1] = line[-1].strip() #removes /n from list
            trainingdata.append(line)

    training_data = split_by_class(trainingdata)
    test_data = split_by_class(testdata)
    print(test_data)
    run_data = training_data
    results = [] #store results of testing data
    for x in test_data["tbd"]:
        #unacc
        classI = prior_class(run_data,"unacc")
        postProbability = likelihood(run_data,"buying",x[0],"unacc") * likelihood(run_data,"maint",x[1],"unacc") * likelihood(run_data,"doors",x[2],"unacc") * \
                          likelihood(run_data,"persons",x[3],"unacc") * likelihood(run_data,"lug_boot", x[4],"unacc") * likelihood(run_data,"safety", x[5],"unacc") 
        result = classI*postProbability
        #acc
        classI2 = prior_class(run_data,"acc")
        postProbability2 = likelihood(run_data,"buying",x[0],"acc") * likelihood(run_data,"maint",x[1],"acc") * likelihood(run_data,"doors",x[2],"acc") *\
                           likelihood(run_data,"persons",x[3],"acc") * likelihood(run_data,"lug_boot", x[4],"acc") * likelihood(run_data,"safety", x[5],"acc") 
        result2 = classI2*postProbability2
        #good
        classI3 = prior_class(run_data,"good")
        postProbability3 = likelihood(run_data,"buying",x[0],"good") * likelihood(run_data,"maint",x[1],"good") * likelihood(run_data,"doors",x[2],"good") *\
                           likelihood(run_data,"persons",x[3],"good") * likelihood(run_data,"lug_boot", x[4],"good") * likelihood(run_data,"safety", x[5],"good")
        result3 = classI3*postProbability3
        #vgood
        classI4 = prior_class(run_data,"vgood")
        postProbability4= likelihood(run_data,"buying",x[0],"vgood") * likelihood(run_data,"maint",x[1],"vgood") * likelihood(run_data,"doors",x[2],"vgood") *\
                          likelihood(run_data,"persons",x[3],"vgood") * likelihood(run_data,"lug_boot", x[4],"vgood") * likelihood(run_data,"safety", x[5],"vgood") 
        result4 = classI4*postProbability4
        if result > result2 and result > result3 and result > result4:
            x.insert(6, "unacc")
            results.append(x)
        elif result2 > result and result2 > result3 and result2 > result4:
            x.insert(6, "acc")
            results.append(x)
        elif result3 > result and result3 > result2 and result3 > result4:
            x.insert(6, "good")
            results.append(x)
        elif result4 > result and result4 > result2 and result4 > result3:
            x.insert(6, "vgood")
            results.append(x)
        else:
            print("two numbers are equal")
   #outputs results into a file
    with open('cartestRESULTS.data', 'w') as f:
        for x in results:
            f.write("%s\n" % x)
    
    #reopening original data to measure accuracy
    original_data = []
    with open("car.data") as file:
        for line in file: 
            line = line.split(",")
            line[-1] = line[-1].strip() #removes /n from list
            original_data.append(line) 
    print("Accuracy: ", acc(results,original_data))


if __name__ == "__main__":
    main()
    
