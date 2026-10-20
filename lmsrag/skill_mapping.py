"""
STEP 1 - Skill Mapping Dictionary.

Maps the skill names that appear in interaction logs (ASSISTments, EdNet, ...) onto the
topic taxonomy of the RAG corpus (MATH / GSM8K / Khan Academy / TOEIC grammar).

Design
------
* The key is the *exact* log label, so the dictionary is the join key between the log
  and the vector store:  log.skill_name --> canonical skill_tag --> chunks with that tag.
* A skill that is a synonym of another log skill points to it through ``alias_of``; both
  resolve to the same ``skill_tag`` (so e.g. "Percents" and "Percent Of" share documents).
* Corpus documents are tagged *from the problem statement* with the regex rules below, so
  no label is invented by an LLM. ``confidence`` records how trustworthy the rule is.
* Log skills that have no textual counterpart in the corpus (Venn diagram, box-and-whisker,
  circle graph ...) are listed in ``UNMAPPED`` with a reason instead of being forced.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import config

FRAC = r"(?:\\d?frac\{[^{}]+\}\{[^{}]+\}|\b\d+/\d+\b)"


def slugify(name: str) -> str:
    """'Addition and Subtraction Fractions' -> 'Addition-and-Subtraction-Fractions'."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    return s


def _e(label, subjects=None, include=(), exclude=(), conf="medium", gsm8k=False,
       levels=(1, 5), aliases=(), note=""):
    return dict(topic_label=label, subjects=list(subjects) if subjects else None,
                include=list(include), exclude=list(exclude), confidence=conf,
                gsm8k=gsm8k, levels=list(levels), aliases=list(aliases), note=note)


PRE, ALG, NT, GEO, CP, IA = ("Prealgebra", "Algebra", "Number Theory", "Geometry",
                             "Counting & Probability", "Intermediate Algebra")

# canonical log skill -> rule.  ORDER = priority when two rules tie (specific first).
CANONICAL: dict[str, dict] = {
    # ---- fractions / decimals / percents ------------------------------------
    "Addition and Subtraction Fractions": _e(
        "Fractions - Addition & Subtraction", [PRE, ALG, NT], gsm8k=True, conf="high",
        include=[FRAC + r"\s*[\+\-]\s*" + FRAC, r"\b(add|subtract|sum|difference)\b[^.]{0,60}\bfractions?\b"]),
    "Multiplication Fractions": _e(
        "Fractions - Multiplication", [PRE, ALG], gsm8k=True,
        include=[FRAC + r"\s*(?:\\cdot|\\times|\*)\s*" + FRAC, r"\b\d+/\d+ of\b",
                 r"\\d?frac\{\d+\}\{\d+\}\s*of\b", r"\bproduct of[^.]{0,30}fractions?"]),
    "Division Fractions": _e(
        "Fractions - Division", [PRE, ALG], conf="high",
        include=[r"\\div\s*" + FRAC, FRAC + r"\s*\\div", r"\breciprocal\b", r"\bdivid\w*[^.]{0,40}fractions?"]),
    "Ordering Fractions": _e(
        "Fractions - Ordering & Comparing", [PRE, ALG],
        include=[r"\b(least|smallest|greatest|largest)\b[^.]{0,40}\bfractions?\b",
                 r"\border\b[^.]{0,40}\bfractions?\b", r"\bcompare\b[^.]{0,30}\bfractions?\b"]),
    "Conversion of Fraction Decimals Percents": _e(
        "Fractions, Decimals & Percents - Conversion", [PRE, ALG, NT], conf="high",
        include=[r"\bas a (?:common )?(?:fraction|decimal|percent)", r"\brepeating decimal",
                 r"\bexpress\b[^.]{0,60}\b(?:fraction|decimal|percent)"]),
    "Finding Percents": _e(
        "Percents - Finding the Percent", [PRE, ALG], gsm8k=True, conf="high",
        include=[r"\bwhat percent", r"\bwhat percentage", r"\bpercent(?:age)?\b[^.]{0,30}\b(?:increase|decrease|change)\b",
                 r"\b(?:increase|decrease)\b[^.]{0,30}\bpercent"]),
    "Percent Discount": _e(
        "Percents - Discounts & Sales", [PRE, ALG], gsm8k=True,
        include=[r"\bdiscount", r"\bmarked down", r"\bsale price", r"\b\d+\s*(?:\\%|%)\s*off\b"]),
    "Percent Of": _e(
        "Percents - Percent of a Quantity", [PRE, ALG], gsm8k=True, conf="high", aliases=["Percents"],
        include=[r"\b\d+(?:\.\d+)?\s*(?:\\%|%|percent)\s+of\b", r"\bpercent of\b"]),
    "Proportion": _e(
        "Ratios & Proportions", [PRE, ALG, GEO], gsm8k=True,
        include=[r"\bproportion", r"\bratio\b", r"\bin the ratio\b", r"\bdirectly proportional"]),
    "Unit Rate": _e(
        "Rates & Unit Rates", [PRE, ALG], gsm8k=True, aliases=["Rate"],
        include=[r"\bunit rate\b", r"\bper (?:hour|minute|gallon|pound|ounce|mile|day|week|second)\b",
                 r"\bmiles per\b", r"\bat a rate of\b"]),
    # ---- equations / expressions ---------------------------------------------
    "Equation Solving More Than Two Steps": _e(
        "Linear Equations - Multi-step", [PRE, ALG], levels=(2, 5), conf="high",
        include=[r"\d*[a-z]\s*[\+\-]\s*\d+\s*=\s*-?\d*[a-z]\b", r"=\s*\d*\(\s*[a-z]", r"\d+\s*\(\s*[a-z]\s*[\+\-]\s*\d+\s*\)\s*[\+\-=]"],
        exclude=[r"\bquadratic", r"\\sqrt", r"\^2"]),
    "Equation Solving Two or Fewer Steps": _e(
        "Linear Equations - One or Two Steps", [PRE, ALG], levels=(1, 3), conf="high",
        include=[r"\$\s*-?\d*[a-z]\s*[\+\-]\s*\d+\s*=\s*-?\d+\s*\$", r"\$\s*-?\d+[a-z]\s*=\s*-?\d+\s*\$",
                 r"\$\s*[a-z]\s*[\+\-]\s*\d+\s*=\s*-?\d+\s*\$", r"\bsolve for \$?[a-z]\$?[\.\?]?\s*$",
                 r"^(?:solve for|find|what is the value of|if)\b[^=$]{0,40}\$[^$=^|\\()]*[a-z][^$=^|\\()]*=[^$=^|\\()a-z]*\$[^$=]{0,25}$",
                 r"\bvalue of \$?[a-z]\$? (?:in|that satisf\w+) the equation \$[^$=^|\\()]*=[^$=^|\\()]*\$"],
        exclude=[r"\bquadratic", r"\\sqrt", r"\^"]),
    "Solving for a variable": _e(
        "Solving for a Variable (literal equations)", [ALG], aliases=["Algebraic Solving"],
        include=[r"\bsolve for \$?[a-z]\$? in terms of", r"\bin terms of\b[^.]{0,20}\$?[a-z]\$?"]),
    "Order of Operations All": _e(
        "Order of Operations", [PRE, ALG], aliases=["Order of Operations +,-,/,* () positive reals"], conf="high",
        include=[r"\border of operations", r"\bpemdas\b",
                 r"^(?:compute|evaluate|calculate|simplify|what is|find)[^a-z$]{0,3}\$(?=[^$]*\()(?=[^$]*(?:\\cdot|\\times|\\div|\*|/))[0-9\s\(\)\+\-\*/\\cdotimesdiv\^\.\{\},]+\$[\.\?]?\s*$"]),
    "Exponents": _e(
        "Exponents & Powers", [PRE, ALG], conf="high",
        include=[r"\bexponents?\b", r"\b(?:evaluate|compute|calculate|simplify|what is)\b[^.$]{0,10}\$\s*\(?-?\d+\)?\^\{?-?\d+\}?[^$a-z]{0,6}\$"]),
    "Square Root": _e(
        "Square Roots", [PRE, ALG], conf="high",
        include=[r"\bsquare roots?\b", r"\b(?:evaluate|compute|calculate|simplify|what is)\b[^.$]{0,10}\$\s*\\sqrt\{?\d+\}?[^$a-z]{0,4}\$"]),
    "Scientific Notation": _e("Scientific Notation", [PRE, ALG], conf="high", include=[r"\bscientific notation"]),
    "Absolute Value": _e(
        "Absolute Value", [PRE, ALG], conf="high",
        include=[r"\babsolute value", r"\\left\|", r"\|\s*-?\d+\s*\|"]),
    "Polynomial Factors": _e(
        "Polynomials - Factoring", [ALG, IA],
        include=[r"\bfactor\b[^.]{0,30}(?:polynomial|expression|completely)", r"\bfactor completely"]),
    "Quadratic Formula to Solve Quadratic Equation": _e(
        "Quadratic Equations", [ALG, IA], include=[r"\bquadratic formula", r"\bsolve\b[^.]{0,30}\bquadratic"]),
    "Distributive Property": _e(
        "Distributive Property & Expanding", [PRE, ALG],
        include=[r"\bdistributive", r"\bexpand\b", r"^(?:simplify|expand)\b[^a-z$]{0,3}\$\s*\d*\s*\(\s*[a-z0-9\s\+\-]+\)\s*[\+\-]\s*\d*\s*\("]),
    "Algebraic Simplification": _e(
        "Simplifying Algebraic Expressions", [PRE, ALG], aliases=["Simplifying Expressions positive exponents"],
        include=[r"^(?:simplify|expand and simplify)\b[^$]{0,10}\$[^$]*[a-z][^$]*\$[\.\?]?\s*$"], exclude=[r"\\sqrt", r"\bfraction"]),
    "Write Linear Equation from Ordered Pairs": _e(
        "Linear Functions - Writing Equations", [ALG], aliases=["Write Linear Equation from Graph", "Write Linear Equation from Situation"],
        include=[r"\bequation of the line", r"\bslope-intercept", r"\bline passing through", r"\bline through the points?"]),
    "Finding Slope From Equation": _e(
        "Linear Functions - Slope", [ALG],
        aliases=["Finding Slope From Situation", "Finding Slope from Ordered Pairs", "Slope"],
        include=[r"\bslope of\b", r"\bthe slope\b"]),
    "Midpoint": _e("Coordinate Geometry - Midpoint", [ALG, GEO], conf="high", include=[r"\bmidpoint"]),
    "Solving Systems of Linear Equations": _e(
        "Systems of Linear Equations", [ALG], aliases=["Solving Systems of Linear Equations by Graphing"], conf="high",
        include=[r"\bsystem of (?:linear )?equations", r"\bsimultaneous"]),
    # ---- geometry -----------------------------------------------------------------
    "Pythagorean Theorem": _e(
        "Pythagorean Theorem", [GEO, PRE, ALG], conf="high",
        include=[r"\bpythagorean", r"\bhypotenuse", r"\bright triangle\b[^.]{0,80}\b(?:length|leg|side)"]),
    "Area Trapezoid": _e("Area - Trapezoid", [GEO, PRE], conf="high", include=[r"\btrapezoid"], exclude=[r"\bperimeter\b"]),
    "Area Parallelogram": _e("Area - Parallelogram", [GEO, PRE], include=[r"\bparallelogram"]),
    "Area Triangle": _e("Area - Triangle", [GEO, PRE], include=[r"\barea of (?:a |the |an )?(?:right |isosceles |equilateral )?triangle"]),
    "Area Circle": _e("Area - Circle", [GEO, PRE], include=[r"\barea of (?:a |the )?circle", r"\barea of (?:a |the )?\w+ ?circle"]),
    "Circumference": _e("Circle - Circumference", [GEO, PRE], conf="high", include=[r"\bcircumference"]),
    "Area Rectangle": _e("Area - Rectangle", [GEO, PRE], include=[r"\barea of (?:a |the )?(?:rectangle|rectangular)", r"\brectangle\b[^.]{0,60}\barea\b"],
                         exclude=[r"\bperimeter\b"]),
    "Perimeter of a Polygon": _e("Perimeter of Polygons", [GEO, PRE], gsm8k=True, include=[r"\bperimeter"]),
    "Volume Cylinder": _e("Volume - Cylinder", [GEO, PRE], conf="high", include=[r"\bcylinder\b[^.]{0,80}\bvolume", r"\bvolume\b[^.]{0,80}\bcylinder"]),
    "Volume Sphere": _e("Volume - Sphere", [GEO, PRE], conf="high", include=[r"\bsphere\b[^.]{0,80}\bvolume", r"\bvolume\b[^.]{0,80}\bsphere"]),
    "Volume Rectangular Prism": _e("Volume - Rectangular Prism", [GEO, PRE], gsm8k=False,
                                   include=[r"\bvolume\b[^.]{0,80}\b(?:box|rectangular|prism|cube)", r"\b(?:box|rectangular solid|prism|cube)\b[^.]{0,80}\bvolume"],
                                   exclude=[r"\bcylinder|\bcone|\bsphere|\bpyramid"]),
    "Surface Area Rectangular Prism": _e("Surface Area - Prisms & Cubes", [GEO, PRE],
                                         include=[r"\bsurface area\b[^.]{0,80}\b(?:box|rectangular|prism|cube)", r"\b(?:box|prism|cube)\b[^.]{0,80}\bsurface area"]),
    "Surface Area Cylinder": _e("Surface Area - Cylinder", [GEO, PRE], include=[r"\bsurface area\b[^.]{0,80}\bcylinder", r"\bcylinder\b[^.]{0,80}\bsurface area"]),
    "Complementary and Supplementary Angles": _e("Angles - Complementary & Supplementary", [GEO, PRE, ALG], conf="high",
                                                 include=[r"\bcomplement", r"\bsupplement"]),
    "Interior Angles Triangle": _e("Angles - Triangle Interior", [GEO, PRE], include=[r"\btriangle\b[^.]{0,60}\bangles?\b[^.]{0,40}\b(?:sum|degrees|measure)"],
                                   exclude=[r"\bpolygon|\bquadrilateral|\bpentagon|\bhexagon"]),
    "Interior Angles Figures with More than 3 Sides": _e("Angles - Polygon Interior", [GEO, PRE], conf="high",
                                                         include=[r"\binterior angles?\b", r"\b(?:pentagon|hexagon|octagon|polygon)\b[^.]{0,60}\bangles?"]),
    "Angles on Parallel Lines Cut by a Transversal": _e("Angles - Parallel Lines & Transversals", [GEO, PRE], conf="high",
                                                        include=[r"\btransversal", r"\bparallel lines?\b[^.]{0,60}\bangles?"]),
    "Calculations with Similar Figures": _e("Similar Figures & Scale Factor", [GEO, PRE], aliases=["Scale Factor"],
                                            include=[r"\bsimilar (?:triangles?|figures?|polygons?|rectangles?)", r"\bscale factor", r"\bscale drawing", r"\bscale model"]),
    "Translations": _e("Transformations - Translation", [GEO, ALG, PRE], include=[r"\btranslat(?:ed|ion|e)\b"]),
    "Reflection": _e("Transformations - Reflection", [GEO, ALG, PRE], include=[r"\breflect(?:ed|ion|s)?\b[^.]{0,30}\b(?:over|across|in|about)\b"]),
    "Rotations": _e("Transformations - Rotation", [GEO, ALG, PRE], include=[r"\brotat(?:ed|ion|e)\b[^.]{0,40}\b(?:degrees|about|around|clockwise)"]),
    # ---- probability / statistics / counting -------------------------------------------
    "Probability of Two Distinct Events": _e(
        "Probability - Compound Events", [CP, PRE, ALG], gsm8k=True, conf="high",
        include=[r"\bprobability\b[^.]{0,120}\b(?:both|and then|two|twice|three|independent|without replacement|at least)\b",
                 r"\b(?:both|two|twice)\b[^.]{0,80}\bprobability\b"]),
    "Probability of a Single Event": _e(
        "Probability - Single Event", [CP, PRE], gsm8k=True, aliases=["D.4.8-understanding-concept-of-probabilities"],
        include=[r"\bprobability\b", r"\bchance of\b", r"\bat random\b"]),
    "Counting Methods": _e(
        "Counting Principles", [CP, PRE, NT], conf="high",
        include=[r"\bhow many (?:different |distinct )?(?:ways|arrangements|outfits|combinations|committees|possible)",
                 r"\bpermutation", r"\bcombination", r"\bchoose\b", r"\barrangements?\b"]),
    "Median": _e("Statistics - Median", [PRE, ALG, CP], conf="high", include=[r"\bmedian\b"]),
    "Mean": _e("Statistics - Mean/Average", [PRE, ALG, CP], gsm8k=True, include=[r"\barithmetic mean\b", r"\bmean\b[^.]{0,30}\b(?:of|is)\b", r"\baverage\b"]),
    "Range": _e("Statistics - Range", [PRE, ALG, CP], include=[r"\bthe range of (?:the )?(?:data|set|numbers|values|scores)", r"\brange of (?:the )?(?:data|set|numbers|scores)"]),
    # ---- number sense ----------------------------------------------------------------------
    "Greatest Common Factor": _e("Number Theory - GCF", [NT, PRE, ALG], conf="high",
                                 include=[r"\bgreatest common (?:factor|divisor)", r"\bgcf\b", r"\bgcd\b"]),
    "Least Common Multiple": _e("Number Theory - LCM", [NT, PRE, ALG], conf="high",
                                include=[r"\bleast common multiple", r"\blcm\b"]),
    "Prime Number": _e("Number Theory - Primes", [NT, PRE], include=[r"\bprime (?:number|factor)", r"\bprime\b"]),
    "Divisibility Rules": _e("Number Theory - Divisibility", [NT, PRE], include=[r"\bdivisible by\b", r"\bmultiples? of\b"]),
    "Rounding": _e("Rounding", [PRE, ALG], gsm8k=True, conf="high", include=[r"\bround(?:ed)? (?:to|it to)\b", r"\bnearest (?:tenth|hundredth|whole|integer|thousandth|ten|hundred)"]),
    "Addition and Subtraction Integers": _e(
        "Integers - Addition & Subtraction", [PRE, ALG], conf="low",
        include=[r"^(?:compute|evaluate|calculate|what is|find)[^a-z$]{0,3}\$\s*-?\d+\s*[\+\-]\s*\(?-\d+\)?\s*\$",
                 r"^(?:compute|evaluate|calculate|what is|find)[^a-z$]{0,3}\$[\s\d\+\-\(\)]*[\+\-][\s\d\+\-\(\)]*\$[\.\?]?\s*$"],
        note="negative-number arithmetic only; whole-number arithmetic is too unspecific in word problems"),
    "Multiplication and Division Integers": _e(
        "Integers - Multiplication & Division", [PRE, ALG], conf="low",
        include=[r"^(?:compute|evaluate|calculate|what is|find)[^a-z$]{0,3}\$\s*-?\d+\s*(?:\\cdot|\\times|\\div|\*|/)\s*\(?-\d+\)?\s*\$"]),
    "Mode": _e("Statistics - Mode", [PRE, ALG, CP], conf="high",
               include=[r"\bmode of (?:the )?(?:given )?(?:data|set|values|numbers)\b",
                        r"\b(?:only )?mode is\b",
                        r"\bmost frequent (?:value|number)\b"]),
    "Unit Conversion Within a System": _e(
        "Measurement - Unit Conversion", [PRE, GEO], conf="medium",
        aliases=["Unit Conversion Within a System"],
        include=[r"\bconvert\b[^.]{0,100}\b(?:inches?|feet|foot|yards?|miles?)\b[^.]{0,100}\b(?:inches?|feet|foot|yards?|miles?)\b",
                 r"\bconvert\b[^.]{0,100}\b(?:millimeters?|centimeters?|meters?|kilometers?)\b[^.]{0,100}\b(?:millimeters?|centimeters?|meters?|kilometers?)\b",
                 r"\bconvert\b[^.]{0,100}\b(?:milligrams?|grams?|kilograms?)\b[^.]{0,100}\b(?:milligrams?|grams?|kilograms?)\b"]),
    "Effect of Changing Dimensions Proportionally": _e(
        "Geometry - Effect of Proportional Dimension Change", [GEO, PRE], conf="high",
        aliases=["Effect of Changing Dimensions of a Shape Prportionally"],
        include=[r"\b(?:dimensions?|side lengths?)\b[^.]{0,80}\b(?:proportionally|scale factor|doubled|tripled)\b",
                 r"\b(?:double|triple|halve)\b[^.]{0,80}\b(?:dimensions?|side lengths?|area|volume)\b"]),
    "Equivalent Fractions": _e(
        "Fractions - Equivalent Fractions", [PRE, ALG], conf="high",
        include=[r"\bequivalent fractions?\b", r"\bwrite\b[^.]{0,40}\bfraction\b[^.]{0,40}\bequivalent\b"]),
    "Ordering Positive Decimals": _e(
        "Decimals - Ordering Positive Decimals", [PRE, ALG], conf="high",
        include=[r"\border\b[^.]{0,50}\b(?:positive )?decimals?\b",
                 r"\b(?:least|greatest|smallest|largest) to (?:greatest|least|largest|smallest)\b[^.]{0,50}\bdecimals?\b"]),
    "Ordering Integers": _e(
        "Integers - Ordering", [PRE, ALG], conf="high",
        include=[r"\border\b[^.]{0,50}\bintegers?\b",
                 r"\b(?:least|greatest|smallest|largest) to (?:greatest|least|largest|smallest)\b[^.]{0,50}\bintegers?\b"]),
    "Ordering Real Numbers": _e(
        "Real Numbers - Ordering", [PRE, ALG], conf="high",
        include=[r"\border\b[^.]{0,50}\breal numbers?\b",
                 r"\b(?:least|greatest|smallest|largest) to (?:greatest|least|largest|smallest)\b[^.]{0,50}\breal numbers?\b"]),
    "Addition Whole Numbers": _e(
        "Whole Numbers - Addition", [PRE], conf="high",
        include=[r"\badd(?:ition of)?\b[^.]{0,50}\bwhole numbers?\b",
                 r"\bsum of\b[^.]{0,50}\bwhole numbers?\b"]),
    "Fraction Of": _e(
        "Fractions - Fraction of a Quantity", [PRE, ALG], conf="medium",
        include=[r"\bwhat fraction of\b",
                 r"\b\d+\s*/\s*\d+\s+of\b",
                 r"\\frac\{[^{}]+\}\{[^{}]+\}\s*of\b"]),
    "Multiplication Whole Numbers": _e(
        "Whole Numbers - Multiplication", [PRE], conf="high",
        include=[r"\bmultiply\b[^.]{0,50}\bwhole numbers?\b",
                 r"\bproduct of\b[^.]{0,50}\bwhole numbers?\b"]),
    "Subtraction Whole Numbers": _e(
        "Whole Numbers - Subtraction", [PRE], conf="high",
        include=[r"\bsubtract(?:ion of)?\b[^.]{0,50}\bwhole numbers?\b",
                 r"\bdifference of\b[^.]{0,50}\bwhole numbers?\b"]),
    "Choose an Equation from Given Information": _e(
        "Algebra - Choose an Equation from Information", [PRE, ALG], conf="high",
        aliases=["Choose an Equation from Given Information"],
        include=[r"\bwhich equation\b[^.]{0,100}\b(?:represents|models|describes)\b",
                 r"\bchoose the equation\b[^.]{0,80}\b(?:represents|models|describes)\b"]),
    "Intercept": _e(
        "Coordinate Geometry - Intercepts", [ALG, GEO], conf="high",
        include=[r"\b[xy]-intercepts?\b", r"\bintercepts? of (?:the )?(?:line|graph|function)\b"]),
    "Linear Equations": _e(
        "Algebra - Linear Equations", [PRE, ALG], conf="medium",
        include=[r"\blinear equations?\b", r"\bsolve (?:the )?linear equation\b"]),
    "Multiplication and Division Positive Decimals": _e(
        "Decimals - Multiplication & Division", [PRE, ALG], conf="high",
        aliases=["Multiplication and Division Positive Decimals"],
        include=[r"\b(?:multiply|divide|multiplication|division)\b[^.]{0,60}\bpositive decimals?\b",
                 r"\bpositive decimals?\b[^.]{0,60}\b(?:multiply|divide|multiplication|division)\b"]),
    "Addition and Subtraction Positive Decimals": _e(
        "Decimals - Addition & Subtraction", [PRE, ALG], conf="high",
        aliases=["Addition and Subtraction Positive Decimals"],
        include=[r"\b(?:add|subtract|addition|subtraction)\b[^.]{0,60}\bpositive decimals?\b",
                 r"\bpositive decimals?\b[^.]{0,60}\b(?:add|subtract|addition|subtraction)\b"]),
    "Area Irregular Figure": _e(
        "Geometry - Area of Composite Figures", [GEO, PRE], conf="high",
        include=[r"\barea of (?:an? |this |that )?(?:irregular|composite) (?:figure|shape|polygon)\b",
                 r"\b(?:irregular|composite) (?:figure|shape|polygon)\b[^.]{0,60}\barea\b"]),
    "Angles - Obtuse, Acute, and Right": _e(
        "Geometry - Classifying Angles", [GEO, PRE], conf="high",
        aliases=["Angles - Obtuse, Acute, and Right"],
        include=[r"\b(?:acute|obtuse|right) angles?\b[^.]{0,50}\b(?:measure|degrees|classify|identify)\b",
                 r"\b(?:measure|classify|identify)\b[^.]{0,50}\b(?:acute|obtuse|right) angles?\b"]),
    "Computation with Real Numbers": _e(
        "Real Numbers - Computation", [PRE, ALG], conf="medium",
        include=[r"\b(?:compute|evaluate|simplify|calculate)\b[^.]{0,60}\breal numbers?\b",
                 r"\bcomputation with real numbers\b"]),
    "Recognize Linear Pattern": _e(
        "Sequences - Recognizing Linear Patterns", [PRE, ALG], conf="high",
        aliases=["Recognize Linear Pattern"],
        include=[r"\blinear pattern\b", r"\blinear sequence\b",
                 r"\bsequence\b[^.]{0,80}\bcommon difference\b"]),
    "Recognize Quadratic Pattern": _e(
        "Sequences - Recognizing Quadratic Patterns", [ALG], conf="high",
        aliases=["Recognize Quadratic Pattern"],
        include=[r"\bquadratic pattern\b", r"\bquadratic sequence\b",
                 r"\bsecond differences?\b"]),
    "Polynomial Vocabulary": _e(
        "Polynomials - Terms, Coefficients & Degree", [ALG, IA], conf="medium",
        aliases=["Parts of a Polyomial, Terms, Coefficient, Monomial, Exponent, Variable"],
        include=[r"\b(?:leading )?coefficient of (?:[a-z]\s+in )?(?:the )?(?:polynomial|term)\b",
                 r"\bcoefficient of [a-z] in (?:this |the )?polynomial\b",
                 r"\bmonomials?\b", r"\b(?:degree|terms?) of (?:the )?polynomial\b"]),
    "Solving Inequalities": _e(
        "Inequalities - Solving", [PRE, ALG], conf="high",
        include=[r"\bsolve\b[^.]{0,60}\binequalit(?:y|ies)\b",
                 r"\binequalit(?:y|ies)\b[^.]{0,60}\b(?:solve|solution|solutions)\b"]),
}

# log skill that has no textual counterpart in MATH / GSM8K
UNMAPPED: dict[str, str] = {
    "Venn Diagram": "diagram/visual items; no textual counterpart in MATH/GSM8K",
    "Box and Whisker": "chart reading; no textual counterpart",
    "Circle Graph": "chart reading; no textual counterpart",
    "Table": "table reading; no textual counterpart",
    "Stem and Leaf Plot": "chart reading; no textual counterpart",
    "Scatter Plot": "chart reading; no textual counterpart",
    "Histogram as Table or Graph": "chart reading; no textual counterpart",
    "Number Line": "visual item",
    "Nets of 3D Figures": "visual item",
    "Reading a Ruler or Scale": "measurement depends on reading a visual scale; no suitable visual corpus is indexed",
    "Interpreting Coordinate Graphs": "graph interpretation depends on visual data; no suitable graph corpus is indexed",
    "Congruence": "too few textual problems",
    "Estimation": "too few textual problems",
    "Pattern Finding": "too generic to tag reliably with rules",
}

# schema example for a second dataset (no corpus is loaded for it in this repo)
EDNET_EXAMPLE = {
    "Part 5 - Incomplete Sentences - Verb Tense": {
        "topic_label": "English Grammar - Present/Past Perfect", "corpus": "TOEIC grammar",
        "status": "schema example only - no EdNet log / TOEIC corpus available in this build"}
}


@dataclass
class SkillMapper:
    """Resolves log skill names and tags corpus text."""
    canonical: dict = field(default_factory=lambda: CANONICAL)
    _compiled: dict = field(default_factory=dict, init=False)
    alias_to_canon: dict = field(default_factory=dict, init=False)

    @staticmethod
    def _key(name: str) -> str:
        return " ".join(name.split()).casefold()

    def __post_init__(self):
        for name, rule in self.canonical.items():
            self._compiled[name] = ([re.compile(p, re.I | re.M) for p in rule["include"]],
                                    [re.compile(p, re.I | re.M) for p in rule["exclude"]])
            self.alias_to_canon[self._key(name)] = name
            for a in rule["aliases"]:
                self.alias_to_canon[self._key(a)] = name

    def canonical_name(self, log_skill: str) -> str | None:
        """Resolve case/whitespace-normalized source labels to a canonical skill name."""
        return self.alias_to_canon.get(self._key(log_skill))

    def unmapped_reason(self, log_skill: str) -> str:
        """Return the documented reason for an unsupported source skill, if available."""
        return next((v for k, v in UNMAPPED.items() if self._key(k) == self._key(log_skill)),
                    "no rule written")

    # --- log side -------------------------------------------------------------------
    def to_tag(self, log_skill: str) -> str | None:
        """log skill name -> skill_tag stored in chunk metadata (None if unmapped)."""
        c = self.canonical_name(log_skill)
        return slugify(c) if c else None

    def confidence(self, log_skill: str) -> str | None:
        c = self.canonical_name(log_skill)
        return self.canonical[c]["confidence"] if c else None

    def topic_label(self, tag: str) -> str:
        for name, r in self.canonical.items():
            if slugify(name) == tag:
                return r["topic_label"]
        return tag

    # --- corpus side ------------------------------------------------------------------
    def tag_text(self, text: str, subject: str | None, level: int | None, is_gsm8k: bool):
        """Return (skill_tag, topic_label, confidence) or (None,None,None)."""
        best, best_score = None, 0
        for name, rule in self.canonical.items():
            if is_gsm8k:
                if not rule["gsm8k"]:
                    continue
            else:
                if rule["subjects"] and subject not in rule["subjects"] and not (subject or "").startswith("khan:"):
                    continue
                if level is not None and not (rule["levels"][0] <= level <= rule["levels"][1]):
                    continue
            inc, exc = self._compiled[name]
            if any(p.search(text) for p in exc):
                continue
            score = sum(1 for p in inc if p.search(text))
            if score > best_score:          # strict '>' keeps the earlier (more specific) rule on ties
                best, best_score = name, score
        if best is None:
            return None, None, None
        r = self.canonical[best]
        return slugify(best), r["topic_label"], r["confidence"]

    # --- export -------------------------------------------------------------------------
    def export(self, log_skills: list[str]) -> dict:
        table = {}
        for s in log_skills:
            c = self.canonical_name(s)
            if c:
                r = self.canonical[c]
                table[s] = {"skill_tag": slugify(c), "topic_label": r["topic_label"],
                            "math_subjects": r["subjects"], "confidence": r["confidence"],
                            "alias_of": None if self._key(c) == self._key(s) else c}
            else:
                table[s] = {"skill_tag": None, "reason": self.unmapped_reason(s)}
        return {"assistments": table, "ednet": EDNET_EXAMPLE}


def write_mapping(log_skills: list[str]) -> SkillMapper:
    m = SkillMapper()
    config.MAPPING_JSON.write_text(json.dumps(m.export(sorted(log_skills)), ensure_ascii=False, indent=2), encoding="utf-8")
    return m
