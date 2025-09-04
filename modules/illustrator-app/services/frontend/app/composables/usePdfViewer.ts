import type { TextMapping, Highlight, ParagraphPosition } from "~/types/common";

export default () => {
  const highlights = reactive<Highlight[]>([]);

  // Perform highlight based on user input
  const performHighlight = (pdfViewer: HTMLElement, query: string) => {
    if (!query || !pdfViewer) return;
    const textLayers = pdfViewer.querySelectorAll(".textLayer");
    textLayers.forEach((layer: Element) => highlightNodeText(layer, query));
  };

  function highlightNodeText(node: Node, query: string | { start: number; end: number }) {
    // console.log("highlighting...", node, query);
    const nodeVal = node.nodeValue || "";

    if (node.nodeType === Node.ELEMENT_NODE) {
      Array.from(node.childNodes).forEach((childNode) => highlightNodeText(childNode, query));
    } else if (node.nodeType === Node.TEXT_NODE) {
      let highlightedHTML = "";
      if (typeof query === "object") {
        const { start, end } = query;
        if (!(start >= 0 && end <= nodeVal.length)) {
          return;
        }
        const prefix = nodeVal.slice(0, start);
        const highlighted = nodeVal.slice(start, end);
        const suffix = nodeVal.slice(end);
        highlightedHTML = `${prefix}<span class='highlight'>${highlighted}</span>${suffix}`;
      }
      else {
        const regex = new RegExp(`${query}`, "gi");
        if (!regex.test(nodeVal)) {
          return;
        }
        highlightedHTML = nodeVal.replace(regex, "<span class='highlight'>$&</span>");
      }
      const fragment = document.createRange().createContextualFragment(highlightedHTML);
      (node as Element).replaceWith(fragment);
    }
  }

  function highlightParagraph(textMappings: TextMapping[], paragraphPosition: ParagraphPosition) {
    const filteredMappings = textMappings
      .filter(
        (mapping) =>
          mapping.start <= paragraphPosition.endPosition &&
        mapping.end >= paragraphPosition.startPosition
      );
    // console.log("highlighting rows...", filteredMappings, paragraphPosition);
    const boxes: Highlight[] = [];
    filteredMappings.forEach((mapping) => {
      const { x, y, width, height } = (mapping.node as HTMLElement).getBoundingClientRect();
      // console.log("combined mapping:", mapping, x, y, width, height);
      boxes.push({ x: x + window.scrollX, y: y + window.scrollY, width, height });
      // highlightNodeText(mapping.node as HTMLElement, {
      //   start: Math.max(0, paragraphPosition.startPosition - mapping.start),
      //   end: Math.min(mapping.end - mapping.start + 1, paragraphPosition.endPosition - mapping.start + 1)
      // });
    });
    // Make one highlight from all the boxes
    const x = Math.min(...boxes.map((box) => box.x));
    const y = Math.min(...boxes.map((box) => box.y));
    const width = Math.max(...boxes.map((box) => box.x + box.width)) - x;
    const height = Math.max(...boxes.map((box) => box.y + box.height)) - y;
    const highlight = { x, y, width, height };
    highlights.push(highlight);
    return highlight as Highlight;
  }

  // Combine text from PDF text layers and map each text's position
  const combineTextAndMap = (pdfViewerEl: HTMLElement) => {
    const textLayers = pdfViewerEl.querySelectorAll(".textLayer");
    let combinedText = "";
    let start = 0;
    const textMappings: TextMapping[] = [];
    textLayers.forEach((layer) => {
      Array.from(layer.childNodes).forEach(node => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          const noteTextOrig = getNodeText(node);
          const nodeText = noteTextOrig.trim();
          if (!nodeText) return;
          textMappings.push({ start, end: start + nodeText.length - 1, node });
          combinedText += nodeText + " ";
          start += nodeText.length + 1;
        }
      });
    });
    combinedText = combinedText.replace(/\s+/g, " ");
    return { combinedText, textMappings };
  };

  function getNodeText(node: Node): string {
    if (node.nodeType === Node.TEXT_NODE) {
      return node.nodeValue || "";
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      return Array.from(node.childNodes).map(getNodeText).join(" ");
    }
    return "";
  }

  const performSearch = (pdfViewer: HTMLElement, query: string) => {
    const { combinedText, textMappings } = combineTextAndMap(pdfViewer);
    const highlights: Highlight[] = [];

    try {
      const paragraphPositions = findParagraphPositions(combinedText, query);
      // console.log("paragraphPositions", combinedText, query, paragraphPositions);
      for (const paragraphPosition of paragraphPositions) {
        if (paragraphPosition.startPosition === -1) {
          console.error("Highlight not found in combined text");
        } else {
          const highlight = highlightParagraph(textMappings, paragraphPosition);
          highlight.text = query;
          highlights.push(highlight);
        }
      }
    } catch (error) {
      console.error("Error:", error);
    }
    return highlights;
  };

  const resetHighlight = () => {
    highlights.length = 0;
  };

  function findParagraphPositions(combinedText: string, paragraph: string) {
    const positions: ParagraphPosition[] = [];
    let start = 0;
    while (start !== -1) {
      start = combinedText.indexOf(paragraph, start);
      if (start !== -1) {
        positions.push({ startPosition: start, endPosition: start + paragraph.length - 1 });
        start += paragraph.length;
      }
    }
    return positions;
  }

  return {
    highlights,
    performHighlight,
    resetHighlight,
    performSearch,
    combineTextAndMap
  };
};
