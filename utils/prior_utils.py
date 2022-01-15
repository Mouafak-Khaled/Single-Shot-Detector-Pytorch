import math
# Itersection of union: Area of overlap / Area of union

class Rectangle():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y

        self.width = width
        self.height = height

        self.left = x
        self.right = x + width

        self.top = y
        self.bottom = y + height

    def area(self):
        return self.height * self.width
    
    def intersection(self, rectangle):
        intersection_amount = 0.0

        x_overlap = max(0.0, min(self.right, rectangle.right) - max(self.left, rectangle.left));
        y_overlap = max(0.0, min(self.bottom, rectangle.bottom) - max(self.top, rectangle.top));
        intersection_amount = x_overlap * y_overlap

        return intersection_amount

    def intersection_of_union(self, rectangle):
        inter = self.intersection(rectangle)
        union = self.area() + rectangle.area() - inter
        iou = inter / union

        return iou

# def main():
#     x1, y1, w1, h1 = 0, 0, 5, 5
#     x2, y2, w2, h2 = 2, 2, 1, 1

#     rect_1 = Rectangle(x1, y1, w1, h1)
#     rect_2 = Rectangle(x2, y2, w2, h2)

#     intersection_amount = rect_1.intersection(rect_2)

#     print(intersection_amount)
#     print(rect_1.intersection_of_union(rect_2))


# if __name__ == "__main__":
#     main()




