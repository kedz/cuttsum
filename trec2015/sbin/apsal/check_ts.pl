#!/usr/bin/perl -w

use strict;

# Check a TREC 2015 Temporal Summarization track submission for various
# common errors:
#      * extra fields
#      * extraneous and misssing topics
#      * invalid retrieved documents (approximate check)
#      * invalid sentence id
#      * invalid time stamps for retrieved events
#      * invalid confidence value
# Messages regarding submission are printed to an error log
#
# Results input file is in the form
#     qid, tid, runid, docno, sid, time, con
#
# where
#
# qid is the event identifier
# tid is the team identifier
# runid is the run identifier
# docno must match the KBA doc-id pattern.
# sid must be a non-negative integer (R+ - {0})
# time is a numeric string ([0-9]+) which when cast as an usigned long
# is between the start and end times of the event,inclusive.
# con is a positive real number.

# Change here put error log in some other directory
my $errlog_dir = ".";

# If more than 25 errors, then stop processing; something drastically
# wrong with the file.
my $MAX_ERRORS = 25; 

# topic{qid} = [start_time, end_time, flag]
# This information was scraped from the topic XML.
my %topic = (
    26, [1358323140, 1359619140, 0],
    27, [1351296000, 1351814400, 0],
    28, [1366771500, 1367635500, 0],
    29, [1361455080, 1362319080, 0],
    30, [1330844400, 1331708400, 0],
    31, [1343596680, 1344028680, 0],
    32, [1347321600, 1348963200, 0],
    33, [1357776000, 1358553600, 0],
    34, [1360972800, 1361318400, 0],
    35, [1365984000, 1366416000, 0],
    36, [1363651200, 1364083200, 0],
    37, [1325149200, 1325754000, 0],
    38, [1365033600, 1365811200, 0],
    39, [1359676800, 1360022400, 0],
    40, [1324501200, 1324933200, 0],
    41, [1358208000, 1359072000, 0],
    42, [1360454400, 1360886400, 0],
    43, [1358380800, 1358812800, 0],
    44, [1334102400, 1334534400, 0],
    45, [1351393200, 1352257200, 0],
    46, [1347321600, 1347753600, 0],
    );

my $results_file;       # input file to be checked (input param)
my $line;           # current input line
my $line_num;           # current input line number
my $errlog;         # file name of error log
my $num_errors;         
my $runid_ = "";                # previous runid
my ($qid, $tid, $runid, $docno, $sid, $time, $con);

my $usage = "Usage: $0 resultsfile\n";
$#ARGV == 0 || die $usage;
$results_file = $ARGV[0];

open RESULTS, "<$results_file" ||
    die "Unable to open results file $results_file: $!\n";

my @path = split "/", $results_file;
my $base = pop @path;
$errlog = $errlog_dir . "/" . $base . ".errlog";
open ERRLOG, ">$errlog" ||
    die "Cannot open error log for writing\n";
$num_errors = 0;
$line_num = 0;

# Process submission file line by line
while ($line = <RESULTS>) {
    $line_num++;
    chomp $line;
    next if ($line =~ /^\s*$/);

    if ($line =~ /^\s*#/) { # pass comments through to output
    next;
    }

    my @fields = split " ", $line;
    
    if (scalar(@fields) == 7) {
       ($qid, $tid, $runid, $docno, $sid, $time, $con) = @fields
    } else {
        &error("Wrong number of fields (expecting 7)");
        exit 255;
    }

    if (!exists $topic{int($qid)}) {
    &error("Unknown topic id: `$qid'");
    next;
    }


    # make sure runid matches the pattern and isn't inconsistent
    if (! $runid_) {    # first line --- remember tag 
        $runid_ = $runid;
        if ($runid_ !~ /^[A-Za-z0-9]{1,15}$/) {
            &error("Run ID `$runid_' is malformed");
            next;
        }
    }
    else { # otherwise just make sure one runid used
        if ($runid ne $runid_) {
            &error("Run ID inconsistent (`$runid' and `$runid_')");
            next;
        }
    }

    # make sure DOCNO known
    # check is only partial (i.e. you can construct cases where
    # check_input won't complain but in fact it is invalid)
    my $docid = "";
    (undef,$docid) = split "-", $docno;
    if ($docid && $docid !~ /^[0-9a-f]{32}$/) {
    &error("Unknown document `$docno'");
    next;
    }

    if ((int($sid) != $sid) || $sid < 0) {
    &error("Sentence identifier must be a non-negative integer, not $sid");
    next;
    }

    # Check the timestamp. Integers, for a 32-bit version of Perl that
    # is to be found on most systems, are wide enough to hold the time
    # stamps. The signed integer range is [-2^53, 2^53]. 
    my $bef = @{$topic{int($qid)}}[0];
    my $aft = @{$topic{int($qid)}}[1];
    # See http://www.perlmonks.org/?node_id=718414 for more info.
    if (int($time) < $bef || int($time) > $aft) {
    &error("Topic $qid has time ($time) out of range [$bef, $aft].");
    next;
    }

    if ($con <= 0.0) {
        &error("Confidence value ($con) must be in R+ - {0}.");
        next;
    }

    # Count the number of sentences retrieved per topic
    @{$topic{int($qid)}}[2]++; 
}

# Check for missing topics
foreach my $k (keys %topic) {
    if (@{$topic{$k}}[2] == 0) {
    &error("topic $k is missing from the results.");
    }
    elsif (@{$topic{$k}}[2] > 1000) {
    &error("too many sentences returned for topic $k");
    }
}

print ERRLOG "Finished processing $results_file\n";
close ERRLOG || die "Close failed for error log $errlog: $!\n";

if ($num_errors) { exit 255; }
exit 0;

# print error message, keeping track of total number of errors
sub error {
   my $msg_string = pop(@_);

    print ERRLOG 
    "run $results_file: Error on line $line_num --- $msg_string\n";

    $num_errors++;
    if ($num_errors > $MAX_ERRORS) {
        print ERRLOG "$0 of $results_file: Quit. Too many errors!\n";
        close ERRLOG ||
        die "Close failed for error log $errlog: $!\n";
    exit 255;
    }
}
