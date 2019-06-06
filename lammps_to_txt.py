if __name__ == "__main__":
    # For command-line runs, build the relevant parser
    import argparse as ap
    parser = ap.ArgumentParser(description='Extract thermo data from LAMMPS log file to a text file.')
    parser.add_argument('-f','--file',default='log.lammps', type=str, help='LAMMPS log file.')
    parser.add_argument('-o', '--output', default='lammps_out.txt', type=str, help='Name for output file.')
    parser.add_argument('-w', '--warmup', default=1, type=int, help='Number of lines to skip for warmup.')
    # Parse the command-line arguments
    args=parser.parse_args()

    file = open(args.file, "rt").read()  # "rt" means "read as text"
    # define start and end substrings and use them as delimiters
    start = "Mbytes"
    end = "Loop"
    s = file[file.find(start) + len(start)+1:file.rfind(end)]

    x = s.splitlines()  # get lines of new data
    ns = '\n'  # connect lines with new line command
    start = args.warmup + 1  # index for start of data
    try:
        s2 = open(args.output,'rt').readlines()
        if s2[0].strip() == x[0].strip():
            f = open(args.output, "a+")
            f.write(ns)
            f.write(ns.join(x[start:len(x)]))
        else:
            newf = open(args.output, "w")
            newf.write(x[0])
            newf.write(ns)
            newf.write(ns.join(x[start:len(x)]))

    except:
        newf = open(args.output, "w")
        newf.write(x[0])
        newf.write(ns)
        newf.write(ns.join(x[start:len(x)]))