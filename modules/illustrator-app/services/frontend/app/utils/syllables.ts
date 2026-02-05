/**
 * Syllable Estimator
 *
 * A fast, simple syllable estimator for English-language words.
 * Provides the estimate() function which returns the estimated number of syllables.
 *
 * TS port of the python-syllables library: https://github.com/prosegrinder/python-syllables
 */

const subSyllables = [
  "cial",
  "tia",
  "cius",
  "cious",
  "uiet",
  "gious",
  "geous",
  "priest",
  "giu",
  "dge",
  "ion",
  "iou",
  "sia$",
  ".che$",
  ".ched$",
  ".abe$",
  ".ace$",
  ".ade$",
  ".age$",
  ".aged$",
  ".ake$",
  ".ale$",
  ".aled$",
  ".ales$",
  ".ane$",
  ".ame$",
  ".ape$",
  ".are$",
  ".ase$",
  ".ashed$",
  ".asque$",
  ".ate$",
  ".ave$",
  ".azed$",
  ".awe$",
  ".aze$",
  ".aped$",
  ".athe$",
  ".athes$",
  ".ece$",
  ".ese$",
  ".esque$",
  ".esques$",
  ".eze$",
  ".gue$",
  ".ibe$",
  ".ice$",
  ".ide$",
  ".ife$",
  ".ike$",
  ".ile$",
  ".ime$",
  ".ine$",
  ".ipe$",
  ".iped$",
  ".ire$",
  ".ise$",
  ".ished$",
  ".ite$",
  ".ive$",
  ".ize$",
  ".obe$",
  ".ode$",
  ".oke$",
  ".ole$",
  ".ome$",
  ".one$",
  ".ope$",
  ".oque$",
  ".ore$",
  ".ose$",
  ".osque$",
  ".osques$",
  ".ote$",
  ".ove$",
  ".pped$",
  ".sse$",
  ".ssed$",
  ".ste$",
  ".ube$",
  ".uce$",
  ".ude$",
  ".uge$",
  ".uke$",
  ".ule$",
  ".ules$",
  ".uled$",
  ".ume$",
  ".une$",
  ".upe$",
  ".ure$",
  ".use$",
  ".ushed$",
  ".ute$",
  ".ved$",
  ".we$",
  ".wes$",
  ".wed$",
  ".yse$",
  ".yze$",
  ".rse$",
  ".red$",
  ".rce$",
  ".rde$",
  ".ily$",
  ".ely$",
  ".des$",
  ".gged$",
  ".kes$",
  ".ced$",
  ".ked$",
  ".med$",
  ".mes$",
  ".ned$",
  ".[sz]ed$",
  ".nce$",
  ".rles$",
  ".nes$",
  ".pes$",
  ".tes$",
  ".res$",
  ".ves$",
  "ere$",
];

const addSyllables = [
  "ia",
  "riet",
  "dien",
  "ien",
  "iet",
  "iu",
  "iest",
  "io",
  "ii",
  "ily",
  ".oala$",
  ".iara$",
  ".ying$",
  ".earest",
  ".arer",
  ".aress",
  ".eate$",
  ".eation$",
  "[aeiouym]bl$",
  "[aeiou]{3}",
  "^mc",
  "ism",
  "^mc",
  "asm",
  "([^aeiouy])1l$",
  "[^l]lien",
  "^coa[dglx].",
  "[^gq]ua[^auieo]",
  "dnt$",
];

const reSubSyllables: RegExp[] = subSyllables.map(pattern => new RegExp(pattern));
const reAddSyllables: RegExp[] = addSyllables.map(pattern => new RegExp(pattern));

/**
 * Estimates the number of syllables in an English-language word
 *
 * @param word - The English-language word to estimate syllables for
 * @returns The estimated number of syllables in the word
 */
export function estimate(word: string): number {
  const lowerWord = word.toLowerCase();
  const parts = lowerWord.split(/[^aeiouy]+/);
  const validParts = parts.filter(part => part !== "");

  let syllables = validParts.length;

  for (const pattern of reSubSyllables) {
    if (pattern.test(lowerWord)) {
      syllables -= 1;
    }
  }

  for (const pattern of reAddSyllables) {
    if (pattern.test(lowerWord)) {
      syllables += 1;
    }
  }

  return syllables <= 0 ? 1 : syllables;
}
