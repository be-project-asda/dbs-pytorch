def getopt(opt, key, default_value):
	if default_value == null and (opt == null or opt[key] == null):
		raise Exception("Error: missing value. Unable to get opt")

	if opt == null:
		return default_value

	v = opt[key]

	if v == null:
		v = default_value

	return v


def dict_average(dictlist):
	sumdict = dict()
	n = 0

	for d in dictlist:
		for k in d.keys():
			sumdict[k] = sumdict[k] + d[k]

	for k in sumdict.keys():
		sumdict[k] = sumdict[k]/len(dictlist)

	return sumdict

def count_keys(d):
	return (len(d.keys()))

def average_values(d):
	sum = 0
	for k in d.keys():
		sum = sum + d[k]

	return sum/(len(d.keys()))